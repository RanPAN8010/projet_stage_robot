import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Génère et sauvegarde trois graphiques (nuage de points, boîtes à moustaches et 
# courbes de densité) pour visualiser la distribution des données selon l'état du conducteur.
def generate_three_french_plots():
    current_path = os.path.abspath(__file__)
    if "edge_server" in current_path:
        base_project_dir = current_path.split("edge_server")[0] + "edge_server"
    else:
        return
        
    data_path = os.path.join(base_project_dir, 'data', 'driver_body_status_train.csv')
    output_dir = os.path.join(base_project_dir, 'data')
    
    if not os.path.exists(data_path):
        print(f"Erreur: Fichier introuvable - {data_path}")
        return
        
    df = pd.read_csv(data_path)
    
    mapping_labels_francais = {
        0: "Normal (Sain)",
        1: "Fatigue Mentale",
        2: "Crise Cardiaque"
    }
    df['État du Conducteur'] = df['Label'].map(mapping_labels_francais)
    
    sns.set_theme(style="whitegrid")
    palette_couleurs = {
        "Normal (Sain)": "#2ecc71",
        "Fatigue Mentale": "#f1c40f",
        "Crise Cardiaque": "#e74c3c"
    }
    
    # ------------------ GRAPHE 1 : NUAGE DE POINTS OPTIMISÉ ------------------
    plt.figure(figsize=(10, 6))
    
    sns.scatterplot(
        data=df,
        x='HeartRate',
        y='HRV',
        hue='État du Conducteur',
        palette=palette_couleurs,
        alpha=0.4,
        s=25,
        edgecolor='none'
    )
    
    q_low_hr, q_hi_hr = df['HeartRate'].quantile(0.005), df['HeartRate'].quantile(0.995)
    q_hi_hrv = df['HRV'].quantile(0.98)
    plt.xlim(q_low_hr - 5, q_hi_hr + 5)
    plt.ylim(-10, q_hi_hrv + 50)
    
    plt.title("Nuage de Points : Cartographie de l'Espace Physiologique", fontsize=13, fontweight='bold', pad=15)
    plt.xlabel("Fréquence Cardiaque (HeartRate - BPM)", fontsize=11, labelpad=8)
    plt.ylabel("Variabilité de la Fréquence Cardiaque (HRV)", fontsize=11, labelpad=8)
    plt.legend(title="État du sujet", loc='upper right')
    plt.tight_layout()
    
    img_path_1 = os.path.join(output_dir, 'distribution_points_physiologique.png')
    plt.savefig(img_path_1, dpi=300)
    plt.close()
    
    # ------------------ GRAPHE 2 : BOÎTES À MOUSTACHES (BOXPLOT) ------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    sns.boxplot(
        ax=axes[0],
        data=df,
        x='État du Conducteur',
        y='HeartRate',
        hue='État du Conducteur',
        palette=palette_couleurs,
        legend=False,
        fliersize=2
    )
    axes[0].set_title("Distribution de la Fréquence Cardiaque", fontsize=11, fontweight='bold', pad=10)
    axes[0].set_xlabel("État du Conducteur", fontsize=10)
    axes[0].set_ylabel("Fréquence Cardiaque (BPM)", fontsize=10)
    
    sns.boxplot(
        ax=axes[1],
        data=df,
        x='État du Conducteur',
        y='HRV',
        hue='État du Conducteur',
        palette=palette_couleurs,
        legend=False,
        fliersize=2
    )
    axes[1].set_ylim(-10, q_hi_hrv + 50)
    axes[1].set_title("Distribution de la Variabilité Cardiaque (HRV)", fontsize=11, fontweight='bold', pad=10)
    axes[1].set_xlabel("État du Conducteur", fontsize=10)
    axes[1].set_ylabel("Variabilité de la Fréquence Cardiaque", fontsize=10)
    
    plt.suptitle("Comparaison par Boîtes à Moustaches des Trois États Évalués", fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    img_path_2 = os.path.join(output_dir, 'comparaison_boites_les_trois_etats.png')
    plt.savefig(img_path_2, dpi=300)
    plt.close()
    
    # ------------------ GRAPHE 3 : HISTOGRAMME ET DENSITÉ (KDEPLOT) ------------------
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    
    sns.histplot(
        ax=axes[0],
        data=df,
        x='HeartRate',
        hue='État du Conducteur',
        palette=palette_couleurs,
        element='step',
        stat='density',
        common_norm=False,
        alpha=0.3,
        kde=True
    )
    axes[0].set_xlim(q_low_hr - 5, q_hi_hr + 5)
    axes[0].set_title("Analyse de Densité de la Fréquence Cardiaque (BPM)", fontsize=11, fontweight='bold', pad=10)
    axes[0].set_xlabel("Fréquence Cardiaque (BPM)", fontsize=10)
    axes[0].set_ylabel("Densité de Probabilité", fontsize=10)
    
    sns.histplot(
        ax=axes[1],
        data=df,
        x='HRV',
        hue='État du Conducteur',
        palette=palette_couleurs,
        element='step',
        stat='density',
        common_norm=False,
        alpha=0.3,
        kde=True
    )
    axes[1].set_xlim(-10, q_hi_hrv + 50)
    axes[1].set_title("Analyse de Densité de la Variabilité Cardiaque (HRV)", fontsize=11, fontweight='bold', pad=10)
    axes[1].set_xlabel("Variabilité de la Fréquence Cardiaque (HRV)", fontsize=10)
    axes[1].set_ylabel("Densité de Probabilité", fontsize=10)
    
    plt.suptitle("Histogrammes et Courbes de Densité de Probabilité", fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    img_path_3 = os.path.join(output_dir, 'analyse_densite_histogramme.png')
    plt.savefig(img_path_3, dpi=300)
    plt.close()
    
    print("==================================================")
    print("Trois types de graphiques générés avec succès !")
    print(f"1. Nuage de points : {img_path_1}")
    print(f"2. Boîtes à moustaches : {img_path_2}")
    print(f"3. Histogrammes de densité : {img_path_3}")
    print("==================================================")

if __name__ == "__main__":
    generate_three_french_plots()