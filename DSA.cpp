#include <bits/stdc++.h>
using namespace std;

typedef long long ll;
typedef vector<ll> vll;
typedef vector<vector<ll>> vvi;
typedef vector<char> vc;
typedef vector<pair<ll, ll>> vp;
typedef vector<bool> vb;
typedef vector<string> vs;

#define all(v) v.begin(), v.end()
#define pb(x) push_back(x)
#define rep_r(i, n, k) for (int i = n - 1; i >= k; i--)
#define repi(i, k, n) for (int i = k; i < n; i++)
#define rep(k, n) for (int i = k; i < n; i++)
#define Yes cout << "Yes\n"
#define per(k, n) for (int i = n - 1; i >= k; i--)
#define YES cout << "YES\n"
#define No cout << "No\n"
#define NO cout << "NO\n"
#define N 100001
#define mod 1000000007
#define Ah 911382323
#define Bh 972663749

string alf = "abcdefghijklmnopqrstuvwxyz";
string alf_M = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
int dx[] = {+0, +0, -1, +1, +1, +1, -1, -1};
int dy[] = {-1, +1, +0, +0, +1, -1, +1, -1};

//Disjoint Set Union
struct DSU
{
    ll n;
    vll p,sz; 

    DSU(ll m){
        rep(0,m){
            p.pb(i);
            sz.pb(1);
        }
        n=m;
    }

    ll setof(ll x)
    {
        while (x != p[x]) x = p[x];
        return x;
    }
    void merge(ll a, ll b)
    {
        a=setof(a);
        b=setof(b);
        if(a==b) return;
        if (sz[a] < sz[b]) swap(a,b);
        {
            p[b] = a;
            sz[a] += sz[b];
        }
        
    }
};

//Sieve de Eratosthenes
vll sieve(){
    vll primes;
    rep(0,N){
        primes.pb(1);
    } 
    primes[0]=0;
    primes[1]=0;
    rep(2,N){
        if(primes[i]==1){
            for(int j=2*i;j<N;j+=i){
                primes[j]++;
            }
        }
    }
    return primes;
}

//Factorization
unordered_map<ll,ll> factorized(ll n){
    unordered_map<ll,ll> mp;
    for(ll i=2;i*i<=n;i++){
        while(n%i==0){
            n/=i;
            mp[i]++;
        }
    }

    if(n>1) mp[n]++;
    
    return mp;
}

//Trie
struct TrieNode {
    char value;
    TrieNode* children[26];
    TrieNode(char value,bool end){
        this->value=value;
        this->end=end;
        rep(0,26){
            children[i]=NULL;
        }
    }
    bool end; 
};
struct Trie{
    ll a;
    TrieNode* root;

    Trie(){
        root=new TrieNode('\0',false);
    }
    
    void insert(string s){
        TrieNode* temp=root;
        rep(0,s.size()){
            if(temp->children[s[i]-'a']==NULL){
                temp->children[s[i]-'a']=new TrieNode(s[i]-'a',false);
            }
            temp=temp->children[s[i]-'a'];
        }
        temp->end=true;
    }

    bool search(string s){
        TrieNode* temp=root;
        rep(0,s.size()){
            if(temp->children[s[i]-'a']==NULL){
                return false;
            }
            temp=temp->children[s[i]-'a'];
        }
        return temp->end;
    }
};

//Fenwick Tree
struct FT{
        vll t;
        FT(ll n){
            t.assign(n+1,0);
        }

        ll suma_(ll b){
            ll sum=0;
            for(ll i=b;i;i-=(i&(-i))){
                sum+=t[i];
            }
            return sum;
        }

        ll suma(ll a,ll b){
            if(a==1)
                return suma_(b);
            return suma_(b)-suma_(a-1);
        }

        void set(ll k,ll value){
            for(ll i=k;i<=t.size();i+=(i&(-i))){
                t[i]+=value;
            }
        }
     };

//Segment Tree (Suma)
// 1 i v: set the element with index i to v (0≤i≤n-1) Ahora si esta bien
// 2 l r: calculate the sum of elements with indices from l to r (0≤l≤r≤n-1) Ahora si esta bien
struct ST
{
    ll sz;
    vll tree;
    ST(ll n,vll &a){
        ll z=ceil(log2(n)),t,y;
        ll x=pow(2,z);
        sz=x;
        tree=vll(2*x);//Cambiar neutro
        rep(0,n){
            tree[x+i]=a[i];
        }
        t=x/2;
        while(t>0){
            y=x;
            repi(i,0,t){
                tree[y/2]=tree[y]+tree[y+1];
                y+=2;
            }
            t/=2;
            x/=2;
        }
    }

    ll sum(ll a, ll b) {
        a += sz; b += sz;
        ll s = 0;//Cambiar neutro
        while (a <= b) {
            if (a%2 == 1) s += tree[a++];
            if (b%2 == 0) s += tree[b--];
            a /= 2; b /= 2;
        }
        return s;
    }

    void set(int k, int x) {
        k += sz;
        tree[k] = x;
        for (k /= 2; k >= 1; k /= 2) {
            tree[k] = tree[2*k]+tree[2*k+1];
        }
    }
};


//Segment Tree Lazy
int tree[4*100001] = {0};  
int lazy[4*100001] = {0};  
 
void updateRangeUtil(int si, int ss, int se, int us, int ue, int diff)
{
    if (lazy[si] != 0)
    {
        tree[si] += (se-ss+1)*lazy[si];
 
        if (ss != se)
        {
            lazy[si*2 + 1]   += lazy[si];
            lazy[si*2 + 2]   += lazy[si];
        }
 
        lazy[si] = 0;
    }
 
    if (ss>se || ss>ue || se<us)
        return ;
 
    if (ss>=us && se<=ue)
    {
        tree[si] += (se-ss+1)*diff;
 
        if (ss != se)
        {
            lazy[si*2 + 1]   += diff;
            lazy[si*2 + 2]   += diff;
        }
        return;
    }
 
    int mid = (ss+se)/2;
    updateRangeUtil(si*2+1, ss, mid, us, ue, diff);
    updateRangeUtil(si*2+2, mid+1, se, us, ue, diff);
 
    tree[si] = tree[si*2+1] + tree[si*2+2];
}
 
void updateRange(int n, int us, int ue, int diff)
{
   updateRangeUtil(0, 0, n-1, us, ue, diff);
}
 
int getSumUtil(int ss, int se, int qs, int qe, int si)
{
    if (lazy[si] != 0)
    {
        tree[si] += (se-ss+1)*lazy[si];
 
        if (ss != se)
        {
            lazy[si*2+1] += lazy[si];
            lazy[si*2+2] += lazy[si];
        }
 
        lazy[si] = 0;
    }
 
    if (ss>se || ss>qe || se<qs)
        return 0;
 
    if (ss>=qs && se<=qe)
        return tree[si];
 
    int mid = (ss + se)/2;
    return getSumUtil(ss, mid, qs, qe, 2*si+1) +
           getSumUtil(mid+1, se, qs, qe, 2*si+2);
}
 
int getSum(int n, int qs, int qe)
{
    if (qs < 0 || qe > n-1 || qs > qe)
    {
        cout <<"Invalid Input";
        return -1;
    }
 
    return getSumUtil(0, n-1, qs, qe, 0);
}
 
void constructSTUtil(int arr[], int ss, int se, int si)
{
    if (ss > se)
        return ;
 
    if (ss == se)
    {
        tree[si] = arr[ss];
        return;
    }
 
    int mid = (ss + se)/2;
    constructSTUtil(arr, ss, mid, si*2+1);
    constructSTUtil(arr, mid+1, se, si*2+2);
 
    tree[si] = tree[si*2 + 1] + tree[si*2 + 2];
}
 
void constructST(int arr[], int n)
{
    constructSTUtil(arr, 0, n-1, 0);
}
 
//Sparse Table Esta Mal ,Tengo que revisarla bien
struct SparseTable
{
    vector<vll> lookup;

    SparseTable(vll &arr, int n)
    {
        lookup.assign(n,vll(ll(log2(n))+1));
        rep(0,n){
            lookup[i][0]=arr[i];
        }

        for (int j = 1; (1 << j) <= n; j++) {
            for (int i = 0; (i + (1 << j) - 1) < n; i++) {
                lookup[i][j]=lookup[i][j - 1]&lookup[i + (1 << (j - 1))][j - 1];
            }
        }
    }
 
    // Returns minimum of arr[L..R] 0<=L,R<n
    int query(int L, int R)
    {
        int j = (int)log2(R - L + 1);
        return lookup[L][j]&lookup[R - (1 << j) + 1][j];
    }
};

//String Hashed
long long compute_hash(string const& s) {
    const int p = 31;
    const int m = 1e9 + 9;
    long long hash_value = 0;
    long long p_pow = 1;
    for (char c : s) {
        hash_value = (hash_value + (c - 'a' + 1) * p_pow) % m;
        p_pow = (p_pow * p) % m;
    }
    return hash_value;
}

vector<int> rabin_karp(string const& s, string const& t) {
    const int p = 31; 
    const int m = 1e9 + 9;
    int S = s.size(), T = t.size();

    vector<long long> p_pow(max(S, T)); 
    p_pow[0] = 1; 
    for (int i = 1; i < (int)p_pow.size(); i++) 
        p_pow[i] = (p_pow[i-1] * p) % m;

    vector<long long> h(T + 1, 0); 
    for (int i = 0; i < T; i++)
        h[i+1] = (h[i] + (t[i] - 'a' + 1) * p_pow[i]) % m; 
    long long h_s = 0; 
    for (int i = 0; i < S; i++) 
        h_s = (h_s + (s[i] - 'a' + 1) * p_pow[i]) % m; 

    vector<int> occurrences;
    for (int i = 0; i + S - 1 < T; i++) {
        long long cur_h = (h[i+S] + m - h[i]) % m;
        if (cur_h == h_s * p_pow[i] % m)
            occurrences.push_back(i);
    }
    return occurrences;
}

//KMP Prefix Function
vector<int> prefix_function(string s) {
    int n = (int)s.length();
    vector<int> pi(n);
    for (int i = 1; i < n; i++) {
        int j = pi[i-1];
        while (j > 0 && s[i] != s[j])
            j = pi[j-1];
        if (s[i] == s[j])
            j++;
        pi[i] = j;
    }
    return pi;
}

//Suffix Array
//Number of Different Substrings
ll number_of_dif_substrings(vector<ll> &lcp){
    ll sum=0;
    rep(0,lcp.size()){
        sum+=lcp[i];
    }
    return (((lcp.size()+2)*(lcp.size()+1))/2)-sum;
}
//Longest Common Prefix (LCP)
vector<ll> lcp_construction(string &s, vector<ll> &p) {
    int n = s.size();
    vector<int> rank(n, 0);
    for (int i = 0; i < n; i++)
        rank[p[i]] = i;

    int k = 0;
    vector<ll> lcp(n-1, 0);
    for (int i = 0; i < n; i++) {
        if (rank[i] == n - 1) {
            k = 0;
            continue;
        }
        int j = p[rank[i] + 1];
        while (i + k < n && j + k < n && s[i+k] == s[j+k])
            k++;
        lcp[rank[i]] = k;
        if (k)
            k--;
    }
    return lcp;
}
//Finding if a substring s is in txt with a suffix array sa of t
ll count_substring(string &s,vll &sa,string &t){

    ll res=0,res2=sa.size()-1;
    rep(0,s.size()){
        ll l=res,r=res2;
        bool h=0;
        while (l <= r) {
            ll mid = (l + r)/2;
            if(sa[mid]+i>=t.size()){
                l=mid+1;
                continue;
            }
            if (s[i]<=t[sa[mid]+i]) {
                res=mid;
                h=1;
                r = mid - 1;
            } 
            else {
                l = mid + 1;
            }
        }
        if(!h) return 0;
        l=res;r=res2;h=0;
        while (l <= r) {
            ll mid = (l + r)/2;
            if(sa[mid]+i>=t.size()){
                l=mid+1;
                continue;
            }
            if (s[i]<t[sa[mid]+i]) {
                r = mid - 1;
            } 
            else {
                h=1;
                res2=mid;
                l = mid + 1;
            }
        }
        if(!h) return 0;
    }
    return res2-res+1;
}
// Structure to store information of a suffix
struct suffix
{
    int index; // To store original index
    int rank[2]; // To store ranks and next rank pair
};
// A comparison function used by sort() to compare two suffixes
// Compares two pairs, returns 1 if first pair is smaller
int cmp(struct suffix a, struct suffix b)
{
    return (a.rank[0] == b.rank[0])? (a.rank[1] < b.rank[1] ?1: 0):
               (a.rank[0] < b.rank[0] ?1: 0);
}
// This is the main function that takes a string 'txt' of size n as an
// argument, builds and return the suffix array for the given string
vll buildSuffixArray(string &txt, int n)
{
    // A structure to store suffixes and their indexes
    struct suffix suffixes[n];
 
    // Store suffixes and their indexes in an array of structures.
    // The structure is needed to sort the suffixes alphabetically
    // and maintain their old indexes while sorting
    for (int i = 0; i < n; i++)
    {
        suffixes[i].index = i;
        suffixes[i].rank[0] = txt[i] - 'a';
        suffixes[i].rank[1] = ((i+1) < n)? (txt[i + 1] - 'a'): -1;
    }
 
    // Sort the suffixes using the comparison function
    // defined above.
    sort(suffixes, suffixes+n, cmp);
 
    // At this point, all suffixes are sorted according to first
    // 2 characters.  Let us sort suffixes according to first 4
    // characters, then first 8 and so on
    int ind[n];  // This array is needed to get the index in suffixes[]
                 // from original index.  This mapping is needed to get
                 // next suffix.
    for (int k = 4; k < 2*n; k = k*2)
    {
        // Assigning rank and index values to first suffix
        int rank = 0;
        int prev_rank = suffixes[0].rank[0];
        suffixes[0].rank[0] = rank;
        ind[suffixes[0].index] = 0;
 
        // Assigning rank to suffixes
        for (int i = 1; i < n; i++)
        {
            // If first rank and next ranks are same as that of previous
            // suffix in array, assign the same new rank to this suffix
            if (suffixes[i].rank[0] == prev_rank &&
                    suffixes[i].rank[1] == suffixes[i-1].rank[1])
            {
                prev_rank = suffixes[i].rank[0];
                suffixes[i].rank[0] = rank;
            }
            else // Otherwise increment rank and assign
            {
                prev_rank = suffixes[i].rank[0];
                suffixes[i].rank[0] = ++rank;
            }
            ind[suffixes[i].index] = i;
        }
 
        // Assign next rank to every suffix
        for (int i = 0; i < n; i++)
        {
            int nextindex = suffixes[i].index + k/2;
            suffixes[i].rank[1] = (nextindex < n)?
                                  suffixes[ind[nextindex]].rank[0]: -1;
        }
 
        // Sort the suffixes according to first k characters
        sort(suffixes, suffixes+n, cmp);
    }
 
    // Store indexes of all sorted suffixes in the suffix array
    vll suffixArr(n);
    for (int i = 0; i < n; i++)
        suffixArr[i] = suffixes[i].index;
 
    // Return the suffix array
    return  suffixArr;
}

//Longest Increasing Subsequence O(N*log(N))
int lengthOfLIS(vector<int>& nums)
{
    int n = nums.size();
    vector<int> ans;
    ans.push_back(nums[0]);
    for (int i = 1; i < n; i++) {
        if (nums[i] > ans.back()) ans.push_back(nums[i]);
        else {
            int low = lower_bound(ans.begin(), ans.end(),nums[i])- ans.begin();
            ans[low] = nums[i];
        }
    }
    return ans.size();
}

//Kadane Subarray SUM_MAX
ll kadane(vll &array){
    ll best = 0, sum = 0;
    rep(0,array.size()){
        sum = max(array[i],sum+array[i]);
        best = max(best,sum);
    }
    return best;
}

//Combinatoria de n en k recursivo en O(n^k)
void combinatoria_n_k(ll i,ll n,ll k,vll &com, vector<vll> &ans){
    if(com.size()==k){
        ans.push_back(com);
        return;
    }
    repi(j,i,n){
        com.push_back(j);
        combinatoria_n_k(j+1,n,k,com,ans);
        com.pop_back();
    }
}

//Inverso Modular en O(log M)
ll mod_inv(ll a, ll m = mod) {
    ll g = m, r = a, x = 0, y = 1;
    while (r != 0) {
        ll q = g / r;
        g %= r;
        swap(g, r);
        x -= q * y;
        swap(x, y);
    }
    return x < 0 ? x + m : x;
}

//Floyd-Warshall (minimum distance between any two nodes)
void floyd_warshall(vector<vll> &distance,ll n, vector<vll> &adj){
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (i == j) distance[i][j] = 0;
            else if (adj[i][j]) distance[i][j] = adj[i][j];
            else distance[i][j] = INT32_MAX;
        }
    }

    for (int k = 0; k < n; k++) {
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                distance[i][j] = min(distance[i][j],distance[i][k]+distance[k][j]);
            }
        }
    }
}

//Dijkstra
void dijkstra(vll &distance,ll x,ll n,vb &seen, vector<vp> &adj){
    priority_queue<pair<ll,ll>> q;
    for (int i = 0; i < n; i++) 
        distance[i] = INT64_MAX;
    distance[x] = 0;
    q.push({0,x});
    while (!q.empty()) {
        ll a = q.top().second; q.pop();
        if (seen[a]) 
            continue;
        seen[a] = true;
        for (auto u : adj[a]) {
            ll b = u.first, w = u.second;
            if (distance[a]+w < distance[b]) {
                distance[b] = distance[a]+w;
                q.push({-distance[b],b});
            }
        }
    }
}

//Prim MST
void Prim_MST(ll x,ll n,vb &seen, vector<vp> &adj,vll &edges){//egdes(n)
    priority_queue<pair<ll,ll>> q;
    q.push({0,x});
    ll o=0;
    while (!q.empty()) {
        ll r = -q.top().first;
        ll a = q.top().second; q.pop();
        if (seen[a]) 
            continue;
        edges[o++]=r;
        seen[a] = true;
        for (auto u : adj[a]) {
            ll b = u.first, w = u.second;
            q.push({-w,b});
        }
    }
}

//MO algotithm (SQRT Descomposition)
struct Query
{
    ll L, R, i;
};
ll block;
bool compare(Query x, Query y)
{
    if (x.L/block != y.L/block)
        return x.L/block < y.L/block;
    return x.R < y.R;
}
void MO(vll &a, ll n, vector<Query> &queries, ll q)
{
    block = (ll)sqrt(n);
 
    sort(all(queries), compare);

    ll currL = 0, currR = 0;
    ll currSum = 0;

    rep(0,q)
    {
        ll L = queries[i].L, R = queries[i].R;
        while (currL < L)
        {
            currSum -= a[currL];
            currL++;
        }
        while (currL > L)
        {
            currL--;
            currSum += a[currL];
        }
        while (currR < R+1)
        {
            currSum += a[currR];
            currR++;
        }
        while (currR > R+1)
        {
            currR--;
            currSum -= a[currR];
        }
    }
}

//Converter binary
ll to_entero(string &s){
	ll sum=0,p=1;
	per(0,s.size()){
		sum+=p*(s[i]-'0');
		p*=10;
	}
	return sum;
}

string to_binario(ll &n){
	string s="";
	while(n!=0){
		ll r=n%2;
		s=to_string(r)+s;
		n/=2;
	}
	return s;
}