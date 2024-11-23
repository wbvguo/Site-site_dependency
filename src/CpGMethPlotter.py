# derived from https://github.com/computational-epigenetics-section/CpGMatrixPlotter.git
# with heavy modificaiton and reimplementation to suit the needs of this project

import numpy as np
import matplotlib.pyplot as plt


class CpGMatrixPlotter:
    def __init__(self, sort_matrix=False):
        self.sort_matrix = sort_matrix

    @staticmethod
    def _buffer_spacings(spacings, radius):
        for i in range(len(spacings)):
            if i == 0:
                continue
            distance = spacings[i] - spacings[i - 1]
            if distance < radius:
                spacings[i] = spacings[i] + (radius - distance)
        return np.array(spacings)

    @staticmethod
    def _get_color(cpg_value: int):
        if cpg_value == 1:
            return "black"
        elif cpg_value == 0:
            return "white"
        else:
            raise NotImplementedError("Value should be either 0, 1, or NaN.")

    @staticmethod
    def format_label(value, precision=2):
        # Check if the value is an integer (either as an int type or a float that is very close to an integer)
        if value.is_integer() or (isinstance(value, float) and value.is_integer()):
            return f"{int(value)}"
        else:
            return f"{value:.{precision}f}"


    def plotCpGMatrix(self, cpgMatrix, cpgPositions, title=None, base_figsize=(12, 8), dpi=300, auto_size=True, expand_ratio=0.1, 
                      title_fontsize=18, xlab_fontsize=10, ylab_fontsize=10, xlab_rot=45, xlab_ha='right', xlab_va='top', 
                      counts=None, count_labels = ['00', '01', '10', '11'], 
                      count_text_offset=0.15, count_text_spacing=0.05, count_label_offset=0.2, count_label_fontsize=12, count_text_fontsize=12,
                      save_path = None):
        
        # Calculate the number of rows and positions
        num_positions = len(cpgPositions)
        num_rows = cpgMatrix.shape[0]

        v_steps = 1 / num_rows
        radius  = min(v_steps / 2.5, 0.02)  # Adjust radius dynamically
        v_spacings = np.arange(0, 1, v_steps)
        h_spacings = (cpgPositions - min(cpgPositions)) / (max(cpgPositions) - min(cpgPositions))
        h_spacings = self._buffer_spacings(h_spacings, radius * 2)

        # Adjust the figure size dynamically based on the number of positions and rows
        figsize_x, figsize_y = base_figsize
        if auto_size:
            ratio_scale = np.max(h_spacings) / np.max(v_spacings)
            figsize_x = np.min([base_figsize[0], base_figsize[1] * ratio_scale])
            figsize_y = figsize_x / ratio_scale
        
        # make the plot
        fig, ax = plt.subplots(figsize=(figsize_x, figsize_y), dpi=dpi)
        if title:
            ax.set_title(title, fontsize=title_fontsize)
        ax.axis("equal")
        # Proportional padding factor
        ax.set_xlim(-expand_ratio, np.max(h_spacings) + expand_ratio)
        ax.set_ylim(-expand_ratio, np.max(v_spacings) + expand_ratio)
        ax.set_xticks(h_spacings)
        ax.set_xticklabels(cpgPositions, rotation=xlab_rot, ha=xlab_ha, va=xlab_va, rotation_mode='anchor', fontsize=xlab_fontsize)
        ax.set_yticks([])
        
        # plot the matrix
        for read, vspace in zip(cpgMatrix, v_spacings):
            ax.axhline(vspace, color="black", zorder=1)
            for cpg, hspace in zip(read, h_spacings):
                if np.isnan(cpg):
                    continue
                circle = plt.Circle((hspace, vspace), radius=radius, facecolor=self._get_color(cpg), edgecolor="black")
                ax.add_artist(circle) 


        # Annotate counts at the bottom of the plot and labels to the left
        if counts is not None:
            midpoints = (h_spacings[:-1] + h_spacings[1:]) / 2 # Calculate the midpoints for each adjacent CpG site
            for row_idx, (row_counts, label) in enumerate(zip(counts, count_labels)):
                for count, midpoint in zip(row_counts, midpoints):
                    # Adjust the text placement to be below the plot area
                    y_position = count_text_offset + count_text_spacing * (row_idx + 1)
                    formatted_label = self.format_label(count)
                    ax.text(midpoint, -y_position, formatted_label, va='top', ha='center', fontsize=count_text_fontsize, zorder=2)
                
                # Annotate the labels to the left of the plot
                ax.text(-count_label_offset, -y_position, label, va='top', ha='right', fontsize=count_label_fontsize, zorder=2)

        plt.tight_layout()
        if save_path is not None:
            plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
        plt.show()

