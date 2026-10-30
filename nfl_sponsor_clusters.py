from sklearn.cluster import KMeans
import pandas as pd

deals = pd.read_csv("NFL_brand_deals_2026-2027.xlsx - 2026-2027.csv")
deals["CATEGORIES"] = deals["CATEGORIES"].replace("Nonprofits, Charities, Associa", "Nonprofits, Charities, Associations")

teams_per_brand = deals.groupby("BRANDS")["TEAM"].nunique()
local_deals = deals[deals["BRANDS"].map(teams_per_brand) <= 3]

df = pd.crosstab(local_deals["TEAM"], local_deals["CATEGORIES"])
df_norm = df.divide(df.sum(axis=1), axis=0)

cluster_model = KMeans(n_clusters=4, random_state=42, n_init=50)



cluster_label = cluster_model.fit_predict(df_norm)
cluster_label

team_cluster_df = pd.DataFrame(cluster_label, index=df_norm.index, columns=["cluster"])

print(team_cluster_df.value_counts())

for cluster,teams in team_cluster_df.groupby("cluster"):
    print(f"cluster: {cluster}, size:{len(teams)}")
    for team in teams.sample(min(5, len(teams))).index:
        print("\t",team)

import matplotlib.pyplot as plt

cluster_names = {0: "Hospitality", 1: "Beer & media", 2: "Industrial", 3: "Giants"}

team = "Seattle Seahawks"
brands = local_deals[(local_deals["TEAM"] == team) & (local_deals["CATEGORIES"] == "Hotel, Restaurant, Leisure")]["BRANDS"]
print(team, list(brands.head(3)))

team = "Pittsburgh Steelers"
peers = team_cluster_df[team_cluster_df["cluster"] == team_cluster_df.loc[team, "cluster"]].index.drop(team)
gap = (df_norm.loc[peers].mean() - df_norm.loc[team]).sort_values(ascending=False)
print(gap.head(5))

for category in ["Hotel, Restaurant, Leisure", "Beverage - Alcohol", "Construction & Industrial"]:
    cluster_share = df_norm[category].groupby(team_cluster_df["cluster"]).mean()
    labels = [cluster_names[c] for c in cluster_share.index] + ["League average"]
    values = list(cluster_share.values * 100) + [df_norm[category].mean() * 100]
    colors = ["#2a78d6"] * len(cluster_share) + ["#a9a8a2"]

    plt.figure(figsize=(8, 4.5))
    plt.bar(labels, values, color=colors)
    plt.xlabel("Cluster")
    plt.ylabel("Share of local deals (%)")
    plt.title(f"{category}: share of local sponsor deals by cluster")
    plt.savefig(f"chart_{category.split()[0].strip(',').lower()}.png", dpi=150, bbox_inches="tight")
    plt.show()