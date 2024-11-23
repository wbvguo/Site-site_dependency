# derived from https://github.com/NuttyLogic/BSBolt/blob/master/bsbolt/CallMethylation/CallVector.py 
# with heavy modificaiton and reimplementation to suit the needs of this project

import numpy as np
import pickle
import pysam
import re
from typing import Any, Dict, List, Tuple, Union


class CallMethVector:
    def __init__(self, bam_file:str=None, genome_db:str=None, contig:str=None, start:int=None, end:int=None, strand:int=None, 
                 cg_only:bool=True, filter_dup:bool=True, fully_contained:bool=True, min_map_qual:int=30, min_base_qual:int=20, 
                 merge_read_pair:bool=True, chunk_size:int=10000, return_queue:bool=None):
        """
        return methylation state vector of methylable sites for each read in the BAM file.

        Arguments:
        - bam_file:         Path to the BAM file.
        - genome_db:        Path to the BSBolt database.
        - contig:           Contig name.
        - start:            Start position of the region of interest.
        - end:              End position of the region of interest.
        - strand:           Strand of the region of interest (1 for Watson, -1 for Crick, 0 for both), default 0.
        - cg_only:          Only report CpG sites, default True.
        - filter_dup:       Filter out duplicate reads, default True.
        - fully_contained:  Only include reads that are fully contained within the region of interest, default True.
        - min_map_qual:     Minimum mapping quality for a read to be included, default 30.
        - min_base_qual:    Minimum base quality to be considered in methylation calling, default 20.
        - merge_read_pair:  Merge the read pair overlapped region (C2T detected in read1, and G2A in read2), default True.
        - chunk_size:       Number of reads to process in each chunk, default 10000.
        - return_queue:     Return the methylation vectors.
        """

        self.input_bam  = pysam.AlignmentFile(bam_file, 'rb', require_index=True)
        self.genome_db  = f'{genome_db}/'        
        self.contig = contig
        self.start  = start
        self.end    = end
        self.strand = strand if strand else 0       # Watson(1) Crick(-1), both(0)
        
        self.cg_only        = cg_only
        self.filter_dup     = filter_dup
        self.fully_contained= fully_contained
        self.min_map_qual   = min_map_qual
        self.min_base_qual  = min_base_qual
        self.merge_read_pair= merge_read_pair if self.strand==0 else False
        self.chunk_size     = chunk_size
        self.return_queue   = return_queue

        self.base_patn_ofst = [('C', 'CG' if self.cg_only else 'C', 0),
                               ('G', 'CG' if self.cg_only else 'G', int(self.cg_only))]
        self.mate_flag_dict = {67: 131, 323: 387, 115: 179, 371: 435, 
                               131: 67, 387: 323, 179: 115, 435: 371}
        self.meth_state_dict= {('C','C'): [True, 1], ('C','T'): [True, 0], 
                               ('G','G'): [True, 1], ('G','A'): [True, 0]}


    def call_meth(self):
        try:
            with open(f'{self.genome_db}{self.contig}.pkl', 'rb') as genome_file:
                chrom_seq = pickle.load(genome_file)
        except FileNotFoundError:
            print(f'{self.contig} not found in BSBolt DB, Methylation call for {self.contig} skipped.'
                  f'Methylation values should be called using the same DB used for alignment.')
            self.return_queue.put([])
        else:
            self.call_contig(chrom_seq)


    def call_contig(self, chrom_seq: str):
        """
        Iterates through bam reads and call methylation vectors for each read. 
        """
        contig_chunk = []
        collect_dict = {}
        
        for read in self.input_bam.fetch(contig=self.contig, start=self.start, end=self.end, multiple_iterators=True):
            if not self.align_filter(read):
                continue    # filter reads based on the read strand, quality or overlapping criterion
            
            ref_base, pattern, offset  = self.base_patn_ofst[1] if read.is_reverse else self.base_patn_ofst[0]
            ref_seq = chrom_seq[read.reference_start - 1: read.reference_end + 2].upper()   # can be optimized
            pos_set = set([match.start() + offset + read.reference_start - 1 for match in re.finditer(pattern, ref_seq)]) # 0-based
            meth_call = self.call_vector(read, pos_set, ref_base)
            if meth_call is None:
                continue
            processed_vector = self.process_meth_vector(read, meth_call, collect_dict)

            if processed_vector:
                contig_chunk.append(processed_vector)
                if len(contig_chunk) >= self.chunk_size:
                    self.return_queue.put((self.contig, contig_chunk))
                    contig_chunk = []

        # process reads that didn't have a pair with a observed methylation site
        for read_dict in collect_dict.values():
            contig_chunk.append((f"{read_dict['name']}/{read_dict['pair_id']}", None, None,
                                 read_dict['call'][:,0], read_dict['call'][:,1], read_dict['call'][:,3],
                                 read_dict['start'], read_dict['end'], read_dict['flag'], read_dict['strand'],
                                 None, None, None, None))
        if contig_chunk:
            self.return_queue.put((self.contig, contig_chunk))


    def align_filter(self, read: pysam.AlignedRead) -> bool:
        """
        Filter reads based on alignment and quality criteria.

        CIGAR string operations:
        0 = M  Alignment match (match or mismatch)
        1 = I  Insertion relative to the reference
        2 = D  Deletion relative to the reference
        3 = N  Skipped region (e.g., introns in spliced alignments)
        4 = S  Soft clipping (read sequence present but not aligned)
        5 = H  Hard clipping (read sequence not present or aligned)
        6 = P  Padding (used for padded alignments, rare)
        7 = =  Sequence match (exact match)
        8 = X  Sequence mismatch

        Returns: True if the read passes all filters, False otherwise.
        """
        if read.is_unmapped:        # filter umapped
            return False
        if self.fully_contained:    # filter partial overlapped
            if read.reference_start > self.end or read.reference_end < self.start:
                return False
        if self.get_read_strand(read) * self.strand < 0:   # filter strand
            return False
        
        is_primary  = not (read.is_secondary or read.is_supplementary)
        is_qualified = (not read.is_qcfail) and read.mapping_quality >= self.min_map_qual
        not_duplicate= not (read.is_duplicate and self.filter_dup)
        not_gap_clip = all(op in (0, 7, 8) for op, length in read.cigar)
        return is_primary and is_qualified and not_duplicate and not_gap_clip


    def call_vector(self, read: pysam.AlignedRead, pos_set: set, ref_base: str) -> List:
        """
        Determine methylation state at specific positions on the read.
        """
        state_list, pos_list, qual_list = [], [], []
        if pos_set:
            ref_cigar_list  = {0, 2, 3, 7, 8}
            query_cigar_list= {0, 1, 4, 7, 8}
            ref_pos, query_pos = read.reference_start, 0
            query_seq, query_qual = read.query_sequence, read.query_qualities

            for cigar_type, cigar_count in read.cigartuples: # cigar_type: operation, cigar_count: length
                if cigar_type in ref_cigar_list and cigar_type in query_cigar_list:
                    # Process positions aligned in both the reference and query
                    for _ in range(cigar_count):
                        pos_qual = query_qual[query_pos]
                        if (ref_pos in pos_set) and (pos_qual > self.min_base_qual):
                            query_base = query_seq[query_pos]
                            call_flag, call_state = self.get_meth_call(ref_base, query_base)
                            if call_flag:
                                state_list.append(call_state)
                                pos_list.append(ref_pos)
                                qual_list.append(pos_qual)
                        ref_pos += 1
                        query_pos += 1
                elif cigar_type in ref_cigar_list:
                    # Process positions aligned in the reference only
                    ref_pos += cigar_count
                elif cigar_type in query_cigar_list:
                    # Process positions aligned in the query only
                    query_pos += cigar_count
        
        return np.column_stack([state_list, pos_list, qual_list, [int(read.is_read2) + 1]*len(state_list)]) if state_list else None


    def process_meth_vector(self, read: pysam.AlignedRead, meth_call: np.ndarray, collect_dict: Dict[str, Any]) -> Union[None, Tuple]:
        """
        Merge methylation calls from paired reads. Handle single reads or paired reads with overlapping regions.
        """
        if self.merge_read_pair and read.is_proper_pair:
            read_label = f'{read.query_name}_{read.flag}_{read.reference_start}'
            mate_label = f'{read.query_name}_{self.mate_flag_dict[read.flag]}_{read.next_reference_start}'
            
            if mate_label in collect_dict:
                # Determine read1 and read2 based on is_read1 flag
                if read.is_read1:
                    read1_dict = {
                        'name': read.query_name,
                        'start': read.reference_start,
                        'end': read.reference_end,
                        'flag': read.flag,
                        'strand': self.get_read_strand(read),
                        'pair_id': int(read.is_read2) + 1,
                        'call': meth_call,
                    }
                    read2_dict = collect_dict.pop(mate_label)
                else:
                    read2_dict = {
                        'name': read.query_name,
                        'start': read.reference_start,
                        'end': read.reference_end,
                        'flag': read.flag,
                        'strand': self.get_read_strand(read),
                        'pair_id': int(read.is_read2) + 1,
                        'call': meth_call,
                    }
                    read1_dict = collect_dict.pop(mate_label)

                # Combine read1 and read2 into a pair
                pair_dict = self.clean_pair(read1_dict, read2_dict)

                return (
                    read.query_name, pair_dict['start'], pair_dict['end'],                      # read_name, fragment start, end
                    pair_dict['call'][:, 0], pair_dict['call'][:, 1], pair_dict['call'][:, 3],  # meth_call, positions, pair_ids
                    read1_dict['start'], read1_dict['end'], read1_dict['flag'], read1_dict['strand'],  # read1 details
                    read2_dict['start'], read2_dict['end'], read2_dict['flag'], read2_dict['strand'],  # read2 details
                )
            else:
                # Store read in the dictionary for later pairing
                collect_dict[read_label] = {
                    'name': read.query_name,
                    'start': read.reference_start,
                    'end': read.reference_end,
                    'flag': read.flag,
                    'strand': self.get_read_strand(read),
                    'pair_id': int(read.is_read2) + 1,
                    'call': meth_call,
                }
                return None

        # Single read case
        return (
            f'{read.query_name}/{int(read.is_read2) + 1}', None, None,
            meth_call[:, 0], meth_call[:, 1], meth_call[:, 3],
            read.reference_start, read.reference_end, read.flag, self.get_read_strand(read),
            None, None, None, None,
        )


    def get_read_strand(self, read: pysam.AlignedRead) -> int:
        """
        Get the strand of the read.
        alterntive method: 
            return {'W': 1, 'C': -1}[read.get_tag('YS')[0]]
        """
        return [1, -1][int(read.is_reverse)]


    def get_meth_call(self, ref_base: str, base_call: str) -> Tuple[bool, Union[None, int]]:
        """
        Detect methylation for C on the sense strand using watson reads, and G (C on the antisense strand) with crick reads
        Arguments:
            ref_base: reference nucleotide
            base_call: read nucleotide
        Returns: (call status, state)
        """
        try:
            tmp_call = self.meth_state_dict[(ref_base, base_call)]
        except:
            return False, None
        else:
            return tmp_call


    def clean_pair(self, read1_dict: Dict, read2_dict: Dict) -> Dict:
        """
        Merge methylation calls from paired reads. For sites in overlapping regions of the read pair, the methylation state with a  
        higher quality score is selected. If overlapping sites have the same quality score, the site of read1 is reported. 
        """
        pos_dict = {row[1]: row for row in read1_dict['call']}  # {pos: [state, pos, qual, pair_id]}
        for row in read2_dict['call']:
            state, pos, qual, pair_id = row
            if pos in pos_dict:
                if state == pos_dict[pos][0]: # if the same call, keep the call with higher quality
                    pos_dict[pos][2] = max(qual, pos_dict[pos][2])
                    pos_dict[pos][3] = 3
                else:
                    if qual > pos_dict[pos][2]:
                        pos_dict[pos]= row
            else:
                pos_dict[pos] = row
        
        clean_call = np.array(list(pos_dict.values()))
        sorting_ix = np.argsort(clean_call[:,1])
        
        return {'start':min(read1_dict['start'], read2_dict['start']), 
                'end':  max(read1_dict['end'], read2_dict['end']), 
                'call': clean_call[sorting_ix,:]}

