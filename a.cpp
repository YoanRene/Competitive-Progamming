#include <bits/stdc++.h>
#include <ext/pb_ds/assoc_container.hpp>
using namespace __gnu_pbds;
using namespace std;


// IMPORT PRACTICE AND CONCENTRATION
// SAVE BEFORE TO SUBMMIT

typedef long long ll;
typedef unsigned long long ull;
typedef vector<ll> vll;
typedef vector<double> vd;
typedef vector<vector<ll>> vvll;
typedef vector<vector<bool>> vvb;
typedef vector<vector<pair<ll, ll>>> vvp;
typedef vector<char> vc;
typedef vector<pair<ll, ll>> vp;
typedef vector<bool> vb;
typedef vector<string> vs;
typedef tree<ll, null_type, less<ll>, rb_tree_tag, tree_order_statistics_node_update> indexed_set;

#define all(v) v.begin(), v.end() //
#define pb(x) push_back(x)
#define per(k, n) for (ll i = n - 1; i >= k; i--)
#define peri(i, k, n) for (ll i = n - 1; i >= k; i--)
#define repi(i, k, n) for (ll i = k; i < n; i++) //
#define rep(k, n) for (ll i = k; i < n; i++)     //

#define Yes cout << "Yes\n"
#define YES cout << "YES\n"
#define No cout << "No\n"
#define NO cout << "NO\n"
#define N 100010
#define mod1 1000000007
#define mod 998244353
#define PI 3.14159265358979323846
#define E 0.00000001
#define SPEED                \
    ios::sync_with_stdio(0); \
    cin.tie(0)

string alf = "abcdefghijklmnopqrstuvwxyz";
string alf_M = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
int dx[] = {+1, +0, -1, +0, };//+1, +1, -1, -1};
int dy[] = {+0, +1, +0, -1, };//+1, -1, +1, -1};

// Call min in set *begin(set_name) ,this is because is the branch most left
// Remember the type of variables
// Dictionary of vectors give us TLE, solution: a vector of vectors with a dictionary that assign to the keys an integer index

// first true
/*bool over = false;
while (l <= r) {
    ll mid = (l + r)/2;
    if (over) {
        res = mid;
        r = mid - 1;
    } else {
        l = mid + 1;
    }
}*/

ll qp(ll b, ll e)
{
    ll ans = 1;
    for (; e; b = (b * b)%mod1 , e >>= 1)
        if (e & 1)
            ans = (ans * b)%mod1 ;
    return ans%mod1;
}

ll im(ll a,ll mod_){
    return qp(a,mod_-2);
}

bool isp(string &s){
    return s[0]==s[3]&&s[1]==s[2];
}

void rec(vector<vll> &v,vll &vari){
    if(vari.size()==3){
        if((vari[0]+vari[1]+vari[2])%10==3){
            v.push_back(vari);
        }
        return;
    }
    rep(0,10){
        vari.push_back(i);
        rec(v,vari);
        vari.pop_back();
    }
}

void solve()
{
    ll n;
    cin>>n;
    vll a(n);

    vll b(n-1);
    rep(0,n){
        cin>>a[i];
    }

    rep(1,n){
        b[i-1]=gcd(a[i],a[i-1]);
    }

    ll c=0, k=-4;
    rep(1,n-1){
        if(b[i-1]>b[i]){
            k=i;
            break;
        }
    }
    vvll b1(3,vll(n-2));
    
    rep(0,3){
        ll t=1;
        repi(j,1,n-1){
            if(j==k)continue;
            b1[i][t-1]=gcd(a[t],a[t-1]);
            t++;
        }
        k++;
    }

    rep(0,3){
        bool f=1;
        repi(j,1,n-2){
            if(b1[i][j]<b1[i][j-1]){
                f=0;
            }
        }
        if(f){
            YES;
            return;
        }
    }
    NO;
   
}


int main()
{
    int test;

    SPEED;
    //freopen("input.txt", "r", stdin);
    //freopen("output.txt", "w", stdout);

    test = 1;
    cin >> test;
    while (test--)
    {
        solve();
    }

    return 0;
}