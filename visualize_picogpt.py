import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle, FancyArrowPatch
import matplotlib.gridspec as gridspec

def visualize_picogpt():
    # Create figure with custom size
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(15, 10))
    gs = gridspec.GridSpec(2, 3, figure=fig)
    
    # Input Sequence
    ax1 = fig.add_subplot(gs[0, 0])
    tokens = ['H', 'E', 'L', 'L', 'O', '>']
    for i, token in enumerate(tokens):
        ax1.add_patch(Rectangle((0, 5-i), 1, 1, fill=True, color='royalblue', alpha=0.3))
        ax1.text(0.5, 5.5-i, token, ha='center', va='center', color='white', fontsize=12)
    ax1.set_title('Input Tokens', pad=20)
    ax1.set_xlim(-0.5, 1.5)
    ax1.set_ylim(-0.5, 6.5)
    ax1.axis('off')

    # Embedding Matrix
    ax2 = fig.add_subplot(gs[0, 1])
    embedding = np.random.rand(6, 8)  # Simplified 6x8 instead of 6x128 for visualization
    im = ax2.imshow(embedding, cmap='coolwarm')
    ax2.set_title('Token Embeddings', pad=20)
    plt.colorbar(im, ax=ax2)
    ax2.set_xlabel('Embedding Dimension')
    ax2.set_ylabel('Token Position')

    # Positional Encoding
    ax3 = fig.add_subplot(gs[0, 2])
    pos_enc = np.zeros((6, 8))
    for pos in range(6):
        for i in range(4):
            pos_enc[pos, 2*i] = np.sin(pos / (10000 ** (2*i/8)))
            pos_enc[pos, 2*i+1] = np.cos(pos / (10000 ** (2*i/8)))
    im = ax3.imshow(pos_enc, cmap='viridis')
    ax3.set_title('Positional Encoding', pad=20)
    plt.colorbar(im, ax=ax3)
    ax3.set_xlabel('Encoding Dimension')
    ax3.set_ylabel('Token Position')

    # Self-Attention Visualization
    ax4 = fig.add_subplot(gs[1, :])
    attention_scores = np.random.rand(6, 6)  # Example attention scores
    im = ax4.imshow(attention_scores, cmap='magma')
    ax4.set_title('Self-Attention Scores', pad=20)
    plt.colorbar(im, ax=ax4)
    ax4.set_xlabel('Key Position')
    ax4.set_ylabel('Query Position')
    
    # Add token labels
    ax4.set_xticks(range(6))
    ax4.set_yticks(range(6))
    ax4.set_xticklabels(tokens)
    ax4.set_yticklabels(tokens)

    # Add explanation text
    fig.text(0.02, 0.02, 
             "PicoGPT Process:\n" +
             "1. Tokenize input\n" +
             "2. Convert to embeddings\n" +
             "3. Add positional encoding\n" +
             "4. Process through transformer blocks\n" +
             "5. Generate output",
             fontsize=10, color='white',
             bbox=dict(facecolor='black', alpha=0.7))

    plt.tight_layout()
    plt.savefig('picogpt_visualization.png', dpi=300, bbox_inches='tight', facecolor='black')
    plt.close()

if __name__ == "__main__":
    visualize_picogpt()
    print("Visualization saved as 'picogpt_visualization.png'") 