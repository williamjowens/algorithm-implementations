import numpy as np
import heapq

class HNSW:
    def __init__(self, data, max_layers, ef_construction, M):
        self.data = np.asarray(data)
        self.max_layers = max_layers
        self.ef_construction = ef_construction
        self.M = M
        self.num_nodes = len(data)
        self.dim = data.shape[1]
        self.layers = [[] for _ in range(max_layers)]
        self._build_index()

    def _build_index(self):
        entry_point = np.random.randint(self.num_nodes)
        self._insert_node(entry_point, 0)
        for i in range(self.num_nodes):
            if i != entry_point:
                self._insert_node(i, 0)

    def _insert_node(self, node_id, layer):
        if layer >= self.max_layers:
            return

        if not self.layers[layer]:
            self.layers[layer].append([node_id])
            self._insert_node(node_id, layer + 1)
            return

        nearest_neighbors = self._search_layer(node_id, self.ef_construction, layer)
        nearest_neighbors.append(node_id)
        self.layers[layer].append(nearest_neighbors)

        if len(self.layers[layer]) > 1 and np.random.rand() < 0.5:
            self._insert_node(node_id, layer + 1)

    def _search_layer(self, node_id, ef, layer):
        nearest_neighbors = []
        visited = set()
        candidates = [(float('inf'), -1)]

        if not self.layers[layer]:
            return []

        entry_point = np.random.choice(len(self.layers[layer]))

        while len(visited) < ef and candidates[0][0] > 0:
            if entry_point in visited:
                heapq.heappop(candidates)
                if not candidates:
                    break
                entry_point = candidates[0][1]
                continue

            visited.add(entry_point)
            distance = self._distance(node_id, self.layers[layer][entry_point][0])
            heapq.heappushpop(candidates, (distance, entry_point))

            for neighbor in self.layers[layer][entry_point]:
                if neighbor not in visited:
                    distance = self._distance(node_id, neighbor)
                    if distance < candidates[0][0]:
                        heapq.heappushpop(candidates, (distance, neighbor))
                        nearest_neighbors.append(neighbor)
                        if len(nearest_neighbors) >= self.M:
                            break

        return nearest_neighbors[:self.M]

    def _distance(self, node_id1, node_id2):
        if isinstance(node_id1, (int, np.integer)):
            node_id1 = self.data[node_id1]
        if isinstance(node_id2, (int, np.integer)):
            node_id2 = self.data[node_id2]
        return np.linalg.norm(node_id1 - node_id2)

    def search(self, query, k):
        query = np.asarray(query)
        if query.ndim == 1:
            query = query.reshape(1, -1)
        if query.shape[1] != self.dim:
            raise ValueError(f"Query shape {query.shape} does not match data dimension {self.dim}")

        results = []
        for q in query:
            nearest_neighbors = self._search_layer(q, self.ef_construction, self.max_layers - 1)
            results.extend(nearest_neighbors)

        distances = [self._distance(q, node_id) for node_id in results]
        sorted_indices = np.argsort(distances)[:k]
        return [results[i] for i in sorted_indices]


def main():
    # Generate random data
    np.random.seed(42)
    data = np.random.rand(1000, 100)

    # Build HNSW index
    hnsw = HNSW(data, max_layers=5, ef_construction=200, M=16)

    # Perform a search
    query = np.random.rand(100)
    k = 5
    nearest_neighbors = hnsw.search(query, k)

    print(f"Nearest neighbors to the query: {nearest_neighbors}")


if __name__ == "__main__":
    main()