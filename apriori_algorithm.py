from collections import defaultdict

class Apriori:
    def __init__(self, min_support):
        self.min_support = min_support

    def fit(self, transactions):
        self.transactions = transactions
        self.itemset_counts = defaultdict(int)
        self.frequent_itemsets = []

        # Generate frequent 1-itemsets
        self.generate_frequent_1_itemsets()

        # Generate frequent k-itemsets for k > 1
        k = 2
        while True:
            candidate_itemsets = self.generate_candidate_itemsets(k)
            if not candidate_itemsets:
                break

            self.count_itemsets(candidate_itemsets)
            frequent_itemsets = self.prune_itemsets(candidate_itemsets)
            if not frequent_itemsets:
                break

            self.frequent_itemsets.extend(frequent_itemsets)
            k += 1

    def generate_frequent_1_itemsets(self):
        itemset_counts = defaultdict(int)
        for transaction in self.transactions:
            for item in transaction:
                itemset_counts[(item,)] += 1

        for itemset, count in itemset_counts.items():
            support = count / len(self.transactions)
            if support >= self.min_support:
                self.itemset_counts[itemset] = support
                self.frequent_itemsets.append(itemset)

    def generate_candidate_itemsets(self, k):
        candidate_itemsets = []
        for itemset1 in self.frequent_itemsets:
            for itemset2 in self.frequent_itemsets:
                if itemset1[:k-2] == itemset2[:k-2] and itemset1[-1] < itemset2[-1]:
                    candidate_itemset = itemset1 + (itemset2[-1],)
                    candidate_itemsets.append(candidate_itemset)
        return candidate_itemsets

    def count_itemsets(self, itemsets):
        itemset_counts = defaultdict(int)
        for transaction in self.transactions:
            for itemset in itemsets:
                if set(itemset).issubset(transaction):
                    itemset_counts[itemset] += 1

        for itemset, count in itemset_counts.items():
            support = count / len(self.transactions)
            self.itemset_counts[itemset] = support

    def prune_itemsets(self, itemsets):
        pruned_itemsets = []
        for itemset in itemsets:
            if self.itemset_counts[itemset] >= self.min_support:
                pruned_itemsets.append(itemset)
        return pruned_itemsets

def main():
    transactions = [
        ['A', 'B', 'C'],
        ['A', 'B', 'D'],
        ['A', 'C', 'D'],
        ['B', 'C', 'D'],
        ['A', 'B', 'C', 'D']
    ]
    min_support = 0.6

    apriori = Apriori(min_support)
    apriori.fit(transactions)

    print(f"Minimum Support: {min_support}")
    print("Frequent Itemsets:")
    for itemset, support in apriori.itemset_counts.items():
        print(f"{itemset}: {support:.2f}")

if __name__ == "__main__":
    main()