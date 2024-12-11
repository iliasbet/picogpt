import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle, FancyArrowPatch
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation
import matplotlib
import torch
import torch.nn.functional as F
from picogpt import PicoGPT, AddPositionalEncoding
import os
import json
from pathlib import Path
matplotlib.use('MacOSX')

class PicoGPTVisualizer:
    def __init__(self):
        self.ipc_dir = Path(".picogpt_viz")
        self.ipc_dir.mkdir(exist_ok=True)
        self.state_file = self.ipc_dir / "model_state.json"
        self.embeddings_file = self.ipc_dir / "embeddings.npy"
        self.attention_file = self.ipc_dir / "attention.npy"
        
        # Set default font sizes and family for better legibility
        plt.rcParams.update({
            'font.size': 9,
            'font.family': 'DejaVu Sans',
            'axes.titlesize': 10,
            'axes.labelsize': 9,
            'xtick.labelsize': 8,
            'ytick.labelsize': 8
        })
        
        # Initialize visualization with smaller size
        plt.style.use('dark_background')
        self.fig = plt.figure(figsize=(12, 8))  # Even smaller size
        self.fig.suptitle("PicoGPT: Learning to Reverse Text!", 
                         fontsize=12, y=0.98, fontweight='bold')
        self.setup_plots()
        
        self.losses = []
        self.current_epoch = 0
        
    def setup_plots(self):
        # Create grid with minimal spacing
        gs = gridspec.GridSpec(3, 6, figure=self.fig, 
                             height_ratios=[1, 1, 1],
                             hspace=0.25, wspace=0.25)
        
        # Input/Output Box
        self.ax1 = self.fig.add_subplot(gs[0, :2])
        self.ax1.set_title('Current Training Example', 
                          pad=5, fontsize=10, fontweight='bold')
        self.setup_input_box()
        
        # Embeddings
        self.ax2 = self.fig.add_subplot(gs[0, 2:5])
        self.im2 = self.ax2.imshow(np.zeros((51, 128)), 
                                  cmap='RdBu_r',
                                  aspect='auto')
        self.ax2.set_title('AI Understanding', 
                          pad=5, fontsize=10, fontweight='bold')
        self.ax2.set_xlabel('Features', fontsize=9)
        self.ax2.set_ylabel('Position', fontsize=9)
        cbar = plt.colorbar(self.im2, ax=self.ax2)
        cbar.set_label('Importance', fontsize=8)
        
        # Attention Matrix
        self.ax3 = self.fig.add_subplot(gs[1, 1:5])
        self.im3 = self.ax3.imshow(np.zeros((51, 51)), 
                                  cmap='magma',
                                  vmin=0, vmax=1)
        self.ax3.set_title('Focus Pattern', 
                          pad=5, fontsize=10, fontweight='bold')
        cbar = plt.colorbar(self.im3, ax=self.ax3)
        cbar.set_label('Attention', fontsize=8)
        
        # Learning Progress
        self.ax4 = self.fig.add_subplot(gs[2, 1:5])
        self.ax4.set_title('Learning Progress', 
                          pad=5, fontsize=10, fontweight='bold')
        self.ax4.set_xlabel('Practice Examples', fontsize=9)
        self.ax4.set_ylabel('Error Rate', fontsize=9)
        self.ax4.grid(True, alpha=0.2, linestyle=':')
        
        # Add explanation boxes
        self.add_explanation_boxes(gs)
        
        plt.tight_layout(rect=[0, 0.02, 1, 0.95])
    
    def add_explanation_boxes(self, gs):
        explanations = [
            (gs[0, 5], "GOAL:\n\nLearning to\nreverse text!\n\nHELLO -> OLLEH"),
            (gs[1, 0], "PROCESS:\n1. Read letters\n2. Learn patterns\n3. Flip word"),
            (gs[1, 5], "COLORS:\nBright = Important\nDark = Less"),
            (gs[2, 0], "NOTE:\nLearning through\npractice!"),
            (gs[2, 5], "PROGRESS:\nLower = Better!")
        ]
        
        box_style = dict(
            facecolor='black',
            edgecolor='white',
            alpha=0.7,
            boxstyle='round,pad=0.5',
            linewidth=1
        )
        
        for pos, text in explanations:
            ax = self.fig.add_subplot(pos)
            ax.text(0.5, 0.5, text,
                   ha='center', va='center',
                   bbox=box_style,
                   fontsize=8,
                   linespacing=1.3,
                   fontweight='bold')
            ax.axis('off')
    
    def draw_text_box(self, text, x, y, label, color):
        """Draw a text box with high contrast"""
        box_style = dict(
            facecolor='black',
            edgecolor=color,
            alpha=0.7,
            boxstyle='round,pad=0.4',
            linewidth=1
        )
        # Draw text
        self.ax1.text(x, y, text,
                     ha='center', va='center',
                     bbox=box_style,
                     fontsize=10,
                     fontweight='bold',
                     color='white')
        # Draw label
        self.ax1.text(x, y + 0.15, label,
                     ha='center', va='bottom',
                     color=color,
                     fontsize=8,
                     fontweight='bold',
                     bbox=dict(facecolor='black', alpha=0.7,
                             edgecolor=color, boxstyle='round,pad=0.2'))
    
    def update_plots(self):
        if not self.state_file.exists():
            return [self.im2, self.im3]
            
        try:
            with open(self.state_file, 'r') as f:
                state = json.load(f)
            
            if state.get('current_sample'):
                self.ax1.clear()
                self.setup_input_box()
                sample = state['current_sample']
                split_idx = sample.find('>')
                input_text = sample[:split_idx]
                target_text = sample[split_idx+1:]
                
                # Draw boxes
                self.draw_text_box(input_text, 0.3, 0.7, "INPUT", "royalblue")
                self.draw_text_box(target_text, 0.3, 0.3, "OUTPUT", "seagreen")
                
                # Add arrow
                arrow = FancyArrowPatch(
                    (0.5, 0.65), (0.5, 0.45),
                    arrowstyle='fancy',
                    color='yellow',
                    mutation_scale=20,
                    linewidth=1.5
                )
                self.ax1.add_patch(arrow)
                self.ax1.text(0.7, 0.55, "REVERSE", 
                            color='yellow', fontsize=9,
                            ha='center', va='center',
                            fontweight='bold',
                            bbox=dict(facecolor='black', alpha=0.7,
                                    edgecolor='yellow', boxstyle='round,pad=0.2'))
            
            # Update visualizations
            if os.path.exists(self.embeddings_file):
                embeddings = np.load(self.embeddings_file)
                self.im2.set_array(embeddings)
            
            if os.path.exists(self.attention_file):
                attention = np.load(self.attention_file)
                self.im3.set_array(attention)
                if state.get('current_sample'):
                    n = max(1, len(sample) // 12)
                    self.ax3.set_xticks(range(0, len(sample), n))
                    self.ax3.set_yticks(range(0, len(sample), n))
                    labels = [sample[i] for i in range(0, len(sample), n)]
                    self.ax3.set_xticklabels(labels, fontsize=7, rotation=45)
                    self.ax3.set_yticklabels(labels, fontsize=7)
            
            # Update progress
            if 'loss' in state:
                self.losses.append(state['loss'])
                self.current_epoch = state.get('epoch', 0)
                
                self.ax4.clear()
                self.ax4.set_title(f'Learning Progress (Try {self.current_epoch})', 
                                 pad=5, fontsize=10, fontweight='bold')
                self.ax4.plot(self.losses, color='cyan', alpha=0.8, linewidth=1.5,
                            marker='.', markersize=2)
                self.ax4.set_xlabel('Practice Examples', fontsize=9)
                self.ax4.set_ylabel('Error Rate', fontsize=9)
                self.ax4.grid(True, alpha=0.2, linestyle=':')
                
                # Add status
                if len(self.losses) > 1:
                    if self.losses[-1] < self.losses[0] * 0.5:
                        msg = "Excellent!"
                    elif self.losses[-1] < self.losses[0] * 0.8:
                        msg = "Better!"
                    else:
                        msg = "Learning..."
                    self.ax4.text(0.02, 0.98, msg,
                                transform=self.ax4.transAxes,
                                fontsize=9, fontweight='bold',
                                bbox=dict(facecolor='black', alpha=0.7,
                                        edgecolor='cyan', boxstyle='round,pad=0.2'))
        
        except (json.JSONDecodeError, FileNotFoundError):
            pass
        
        return [self.im2, self.im3]

    def setup_input_box(self):
        self.ax1.set_xlim(-0.5, 1.5)
        self.ax1.set_ylim(-0.5, 51.5)
        self.ax1.axis('off')
    
    def animate(self, frame):
        return self.update_plots()
    
    def run(self):
        anim = FuncAnimation(self.fig, self.animate, frames=None,
                           interval=100, blit=False)
        plt.show()
    
    def cleanup(self):
        if self.ipc_dir.exists():
            for file in self.ipc_dir.glob("*"):
                file.unlink()
            self.ipc_dir.rmdir()

if __name__ == "__main__":
    visualizer = PicoGPTVisualizer()
    try:
        visualizer.run()
    finally:
        visualizer.cleanup()
    print("Visualization ended") 