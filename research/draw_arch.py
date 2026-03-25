import matplotlib.pyplot as plt

def draw_mapping():
    classes = ['Fight', 'Jump', 'Dance', 'Hug']
    base_gcn = [0.6, 0.7, 0.4, 0.5]
    with_heuristics = [0.92, 0.88, 0.82, 0.75] # Примерный рост точности
    
    x = range(len(classes))
    plt.figure(figsize=(8, 5))
    plt.bar(x, with_heuristics, width=0.4, label='Semantic Scoring', color='#03a9f4')
    plt.bar(x, base_gcn, width=0.4, label='Base GCN', color='lightgray', alpha=0.7)
    
    plt.xticks(x, classes)
    plt.ylabel('Confidence / F1 Estimate')
    plt.title('Impact of Semantic Scoring Engine', fontsize=12, fontweight='bold')
    plt.legend()
    plt.savefig('mapping_impact.png', dpi=300)
    plt.show()

draw_mapping()