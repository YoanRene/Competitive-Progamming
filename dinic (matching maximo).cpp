#include <bits/stdc++.h>
using namespace std;

template <typename T> struct dinic {
	struct edge {
		int src, dst;
		T cap, flow;
		int rev;
	};

	int n;
	vector<vector<edge>> adj;

	dinic(int n) : n(n), adj(n) {}

	void add_edge(int src, int dst, T cap) {
		adj[src].push_back({src, dst, cap, 0, (int)adj[dst].size()});
		if (src == dst)
			adj[src].back().rev++;
		adj[dst].push_back({dst, src, 0, 0, (int)adj[src].size() - 1});
	}

	vector<int> level, iter;

	T augment(int u, int t, T cur) {
		if (u == t)
			return cur;
		for (int &i = iter[u]; i < (int)adj[u].size(); ++i) {
			edge &e = adj[u][i];
			if (e.cap - e.flow > 0 && level[u] > level[e.dst]) {
				T f = augment(e.dst, t, min(cur, e.cap - e.flow));
				if (f > 0) {
					e.flow += f;
					adj[e.dst][e.rev].flow -= f;
					return f;
				}
			}
		}
		return 0;
	}

	int bfs(int s, int t) {
		level.assign(n, n);
		level[t] = 0;
		queue<int> Q;
		for (Q.push(t); !Q.empty(); Q.pop()) {
			int u = Q.front();
			if (u == s)
				break;
			for (int i = 0; i < (int)adj[u].size(); ++i) {
				edge &e = adj[u][i];
				edge &erev = adj[e.dst][e.rev];
				if (erev.cap - erev.flow > 0 && level[e.dst] > level[u] + 1) {
					Q.push(e.dst);
					level[e.dst] = level[u] + 1;
				}
			}
		}
		return level[s];
	}

	const T oo = numeric_limits<T>::max();

	T max_flow(int s, int t) {
		for (int u = 0; u < n; ++u) // initialize
			for (int i = 0; i < (int)adj[u].size(); ++i) {
				edge &e = adj[u][i];
				e.flow = 0;
			}

		T flow = 0;
		while (bfs(s, t) < n) {
			iter.assign(n, 0);
			for (T f; (f = augment(s, t, oo)) > 0;)
				flow += f;
		} // level[u] == n ==> s-side
		return flow;
	}
};

// Resolvamos el problema de matching maximo en un grafo bipartito con flujo
int32_t main() {
	ios_base::sync_with_stdio(0);
	cin.tie(0);

	// Los tamaños de los dos conjuntos de la particion
	int n, m;
	cin >> n >> m;
	dinic<int> g(2 + n + m);
	int s = n + m;     // el source
	int t = n + m + 1; // el sink

	// el nodo de la red de flujo correspondiente al i-esimo nodo de la primera
	// mitad
	auto primera_mitad = [&](int i) {
		return i - 1;
	};

	// el nodo de la red de flujo correspondiente al i-esimo nodo de la segunda
	// mitad
	auto segunda_mitad = [&](int i) {
		return i + n - 1;
	};

	// Las aristas
	int e;
	cin >> e;
	for (int i = 0; i < e; i++) {
		int u, v;
		// se esperan los valores 1-indexed
		cin >> u >> v;
		g.add_edge(primera_mitad(u), segunda_mitad(v), 1);
	}
	for (int i = 1; i <= n; i++)
		g.add_edge(s, primera_mitad(i), 1);
	for (int i = 1; i <= m; i++)
		g.add_edge(segunda_mitad(i), t, 1);

	int cardinalidad_matching = g.max_flow(s, t);

	cout << "La cardinalidad del mayor matching es: " << cardinalidad_matching
	     << "\n";
	cout << "El matching es:\n";
	for (int i = 1; i <= n; i++) {
		// chequeamos las aristas que salen de los nodos de la primera mitad
		for (auto e : g.adj[primera_mitad(i)]) {
			if (e.dst >= n && e.dst < n + m && e.flow == 1) {
				cout << "(" << i << ", " << e.dst - n + 1 << ")\n";
			}
		}
	}
}
