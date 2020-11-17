#include <iostream>
#include <utility>
#include <vector>
#include <fstream>
#include <unordered_set>
#include <algorithm>
#include <numeric>
#include <tgmath.h>

using namespace std;
using ll = long long int;
using ld = long double;
struct table_row {
    ll number;
    vector<ll> features;
    ll y;

};
ll features_num_input, class_num_input, depth_num_input, object_num_input;


vector<table_row> table;

struct state {
    vector<ll> number_of_every_class;
    ll number_of_all_objects;

    explicit state(vector<ll> numberOfEveryClass) :
            number_of_every_class(std::move(numberOfEveryClass)),
            number_of_all_objects(accumulate(number_of_every_class.begin(), number_of_every_class.end(), 0)) {}

    state(vector<ll> numberOfEveryClass, ll numberOfAllObjects) :
            number_of_every_class(std::move(numberOfEveryClass)),
            number_of_all_objects(numberOfAllObjects) {}

    state(ll size_of_empty_vector) :
            number_of_every_class(vector<ll>(size_of_empty_vector, 0)),
            number_of_all_objects(0) {}

    state(state const &other) = default;

    void add_row(ll row) {
        ++number_of_every_class[table[row].y];
        ++number_of_all_objects;
    }

    void remove_row(ll row) {
        --number_of_every_class[table[row].y];
        --number_of_all_objects;
    }

    size_t size_of_class_number_vector() {
        return number_of_every_class.size();
    }
};

using calc_energy_func_t = ld (*)(state &);


ld giny_impurity(state &st) {
    ld sum = 0;
    for (ll num : st.number_of_every_class) {
        ld t= (ld) num / st.number_of_all_objects;
        sum += t * t;
    }
    return 1 - sum;
//    vector<ld> sqr_frequencies;
//    transform(st.number_of_every_class.begin(), st.number_of_every_class.end(), back_inserter(sqr_frequencies),
//              [st](ll i) {
//                  ld temp = ((ld) i / st.number_of_all_objects);
//                  return temp * temp;
//              });
//    ld sum_of_sqrs = accumulate(sqr_frequencies.begin(), sqr_frequencies.end(), (ld) 0);
//    return 1 - sum_of_sqrs;
}

ld iginy(state &st) {
    vector<ld> logs;
    transform(st.number_of_every_class.begin(), st.number_of_every_class.end(), back_inserter(logs),
              [st](ll i) {

                  ld temp = ((ld) i / st.number_of_all_objects);
                  if (temp != 0) {
                      return temp * log2l(temp);
                  }
                  return (ld) 0;

              });
    ld sum_of_logs = accumulate(logs.begin(), logs.end(), (ld) 0);
    return -sum_of_logs;
}

calc_energy_func_t calc_energy = giny_impurity;

struct node {
    node *left = nullptr;
    node *right = nullptr;
    ll num_feature = -1;
    ld separate_value = 0;
    ll leaf_class = -1;
    ll level = 0;
    ll index;

    node() = default;

    node(ll level) : level(level) {}

    ~node() {
        delete left;
        delete right;
    }

    void separate_node(state &general_state, vector<ll> &rows) {
        if (level > depth_num_input) {
            create_leaf(general_state);
            return;
        }
        ld current_energy = calc_energy(general_state);

        ll min_feature = -1;
        ld min_separate_value = 0;
        state min_left_state(general_state);
        state min_right_state(general_state.size_of_class_number_vector());
        vector<ll> min_left_rows;
        vector<ll> min_right_rows;

        for (ll i = 0; i < features_num_input; i++) {
            sort(rows.begin(), rows.end(), [i](ll row_num_1, ll row_num_2) {
                return table[row_num_1].features[i] < table[row_num_2].features[i];
            });
            state left_state(general_state.size_of_class_number_vector());
            state right_state(general_state);
            for (ll j = 0; j < (rows.size() - 1); j++) {
                left_state.add_row(rows[j]);
                right_state.remove_row(rows[j]);
                ld left_energy = calc_energy(left_state);
                ld right_energy = calc_energy(right_state);
                ld weighted_energy =
                        left_state.number_of_all_objects / (ld) general_state.number_of_all_objects *
                        left_energy +
                        right_state.number_of_all_objects / (ld) general_state.number_of_all_objects *
                        right_energy;
                if (weighted_energy < current_energy) {

                    current_energy = weighted_energy;
                    min_feature = i;
                    min_separate_value = ((ld) table[rows[j]].features[i] + table[rows[j + 1]].features[i]) / 2;

                    min_left_state = left_state;
                    min_right_state = right_state;

                    min_left_rows.clear();
                    min_right_rows.clear();
                    for (ll k = 0; k < rows.size(); k++) {
                        if (k <= j) {
                            min_left_rows.push_back(rows[k]);
                        } else {
                            min_right_rows.push_back(rows[k]);
                        }
                    }

                }
            }
        }
        if (min_feature != -1) {
            left = new node(level + 1);
            right = new node(level + 1);
            separate_value = min_separate_value;
            num_feature = min_feature;
            left->separate_node(min_left_state, min_left_rows);
            right->separate_node(min_right_state, min_right_rows);
        } else {
            create_leaf(general_state);
        }
    }

    void create_leaf(state &st) {
        ll max_class_count = st.number_of_every_class[1];
        ll max_class = 1;

        for (ll i = 1; i < st.number_of_every_class.size(); i++) {
            if (st.number_of_every_class[i] > max_class_count) {
                max_class_count = st.number_of_every_class[i];
                max_class = i;
            }
        }
        leaf_class = max_class;
    }

    ll init_indexes(ll cur_num) {//получаю то, чем могу стать сам, возващаю то, чем может стать другой
        if (is_leaf()) {
            index = cur_num;
            return cur_num + 1;
        }
        index = cur_num;
        ll next_ind = left->init_indexes(index + 1);
        next_ind = right->init_indexes(next_ind);
        return next_ind;
    }

    bool is_leaf() const {
        return num_feature == -1;
    }

    void count_nodes(ll &counter) {
        ++counter;
        if (!is_leaf()) {
            left->count_nodes(counter);
            right->count_nodes(counter);
        }
    }

};

ostream &operator<<(ostream &os, const node &dt) {
    if (dt.is_leaf()) {
        os << "C " << dt.leaf_class << "\n";
    } else {
        os << "Q " << (dt.num_feature + 1) << " " << dt.separate_value << " " << dt.left->index << " "
           << dt.right->index
           << "\n" << (*dt.left) << (*dt.right);
    }
    return os;
}

int main() {
#ifdef MY_DEBUG
    ifstream cin("a.in");
    ofstream cout("a.out");
#endif
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);
    cout.tie(nullptr);

    cin >> features_num_input >> class_num_input >> depth_num_input >> object_num_input;
    state st(class_num_input + 1);
    vector<ll> rows;

    for (ll i = 0; i < object_num_input; i++) {
        table_row row{i};
        for (ll i = 0; i < features_num_input; i++) {
            ll t;
            cin >> t;
            row.features.push_back(t);
        }
        cin >> row.y;
        table.push_back(row);
        st.add_row(i);
        rows.push_back(i);
    }
    if (object_num_input < 1000) {
        calc_energy = iginy;
    }

    node root(1);
    vector<table_row> &gl = table;
    root.separate_node(st, rows);
    root.init_indexes(1);
    ll number_nodes = 0;
    root.count_nodes(number_nodes);
    cout << number_nodes << "\n" << root;
}