//-------------------------------------------------------------------//
//         eduPIC : educational 1d3v PIC/MCC simulation code         //
//           version 1.0, release date: March 16, 2021               //
//                       :) Share & enjoy :)                         //
//-------------------------------------------------------------------//
// When you use this code, you are required to acknowledge the       //
// authors by citing the paper:                                      //
// Z. Donko, A. Derzsi, M. Vass, B. Horvath, S. Wilczek              //
// B. Hartmann, P. Hartmann:                                         //
// "eduPIC: an introductory particle based  code for radio-frequency //
// plasma simulation"                                                //
// Plasma Sources Science and Technology, vol 30, pp. 095017 (2021)  //
//-------------------------------------------------------------------//
// Disclaimer: The eduPIC (educational Particle-in-Cell/Monte Carlo  //
// Collisions simulation code), Copyright (C) 2021                   //
// Zoltan Donko et al. is free software: you can redistribute it     //
// and/or modify it under the terms of the GNU General Public License//
// as published by the Free Software Foundation, version 3.          //
// This program is distributed in the hope that it will be useful,   //
// but WITHOUT ANY WARRANTY; without even the implied warranty of    //
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU  //
// General Public License for more details at                        //
// https://www.gnu.org/licenses/gpl-3.0.html.                        //
//-------------------------------------------------------------------//



#include <random>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <iterator>
#include <array>
#include <vector>
#include <algorithm>
#include <numeric>
#include <string>
#include <sstream>
#include <cstdlib>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <mpi.h>

using namespace std;

// constants

const double     PI             = 3.141592653589793;      // mathematical constant Pi
const double     TWO_PI         = 2.0 * PI;               // two times Pi
const double     E_CHARGE       = 1.60217662e-19;         // electron charge [C]
const double     EV_TO_J        = E_CHARGE;               // eV <-> Joule conversion factor
const double     E_MASS         = 9.10938356e-31;         // mass of electron [kg]
const double     AR_MASS        = 6.63352090e-26;         // mass of argon atom [kg]
const double     MU_ARAR        = AR_MASS / 2.0;          // reduced mass of two argon atoms [kg]
const double     K_BOLTZMANN    = 1.38064852e-23;         // Boltzmann's constant [J/K]
const double     EPSILON0       = 8.85418781e-12;         // permittivity of free space [F/m]

// simulation parameters (runtime-configurable defaults)

int        N_G            = 400;                    // number of grid points
int        N_T            = 4000;                   // time steps within an RF period
double     FREQUENCY      = 13.56e6;                // driving frequency [Hz]
double     VOLTAGE        = 250.0;                  // voltage amplitude [V]
double     L              = 0.025;                  // electrode gap [m]
double     PRESSURE       = 10.0;                   // gas pressure [Pa]
double     TEMPERATURE    = 350.0;                  // background gas temperature [K]
double     WEIGHT         = 7.0e4;                  // weight of superparticles
double     ELECTRODE_AREA = 1.0e-4;                 // (fictive) electrode area [m^2]
int        N_INIT         = 1000;                   // number of initial electrons and ions
int        N_SUB          = 20;                     // ions move only in these cycles (subcycling)
int        N_BIN          = 20;                     // number of time steps binned for the XT distributions
double     GAMMA_I        = 0.07;                   // ion-induced secondary electron emission probability
double     GAMMA_E        = 0.0;                    // electron reflection probability at electrodes
int        N_IANG         = 90;                     // number of angle bins for ion-wall impact angles
double     DE_IANG        = 1.0;                    // angle bin size for ion-wall impact angles [deg]
bool       USE_BOLTZMANN_ELECTRONS = false;         // use Boltzmann electrons for density (semi-analytic)
int        NE_CENTER_MODE = 0;                      // 0=off, 1=fixed, 2=table
double     NE_CENTER_FIXED = 1.0e15;                // center electron density [1/m^3] when NE_CENTER_MODE=1
double     TE_INIT_EEV = 3.0;                       // initial electron temperature [eV] for Boltzmann density
double     TE_MIN_EEV  = 0.1;                       // floor for electron temperature [eV]
int        BOLTZMANN_ITER = 8;                      // fixed-point iterations for Boltzmann-Poisson
double     BOLTZMANN_DAMP = 0.6;                    // under-relaxation factor (0,1]
string     NE_CENTER_TABLE = "Time-dependent-ne.csv"; // time table for center density

// additional (derived) constants

double     PERIOD         = 0.0;                   // RF period length [s]
double     DT_E           = 0.0;                   // electron time step [s]
double     DT_I           = 0.0;                   // ion time step [s]
double     DX             = 0.0;                   // spatial grid division [m]
double     INV_DX         = 0.0;                   // inverse of spatial grid size [1/m]
double     GAS_DENSITY    = 0.0;                   // background gas density [1/m^3]
double     OMEGA          = 0.0;                   // angular frequency [rad/s]
int        N_XT           = 0;                     // number of spatial bins for the XT distributions

// electron and ion cross sections

const int        N_CS           = 5;                      // total number of processes / cross sections
const int        E_ELA          = 0;                      // process identifier: electron/elastic
const int        E_EXC          = 1;                      // process identifier: electron/excitation
const int        E_ION          = 2;                      // process identifier: electron/ionization
const int        I_ISO          = 3;                      // process identifier: ion/elastic/isotropic
const int        I_BACK         = 4;                      // process identifier: ion/elastic/backscattering
const double     E_EXC_TH       = 11.5;                   // electron impact excitation threshold [eV]
const double     E_ION_TH       = 15.8;                   // electron impact ionization threshold [eV]
const int        CS_RANGES      = 1000000;                // number of entries in cross section arrays
const double     DE_CS          = 0.001;                  // energy division in cross section arrays [eV]
using cross_section = array<float,CS_RANGES>;             // cross section array
 
cross_section    sigma[N_CS];                             // set of cross section arrays
cross_section    sigma_tot_e;                             // total macroscopic cross section of electrons
cross_section    sigma_tot_i;                             // total macroscopic cross section of ions

// particle coordinates

size_t     N_e = 0;                                       // number of electrons (local per MPI rank)
size_t     N_i = 0;                                       // number of ions (local per MPI rank)

vector<double>  x_e, vx_e, vy_e, vz_e;                    // coordinates of electrons (one spatial, three velocity components)
vector<double>  x_i, vx_i, vy_i, vz_i;                    // coordinates of ions (one spatial, three velocity components)

using xvector = vector<double>;                           // vector for quantities defined at grid points
xvector  efield, pot;                                     // electric field and potential
xvector  e_density, i_density;                            // electron and ion densities
xvector  cumul_e_density, cumul_i_density;                // cumulative densities

using Ullong = unsigned long long int;                    // compact name for 64 bit unsigned integer
Ullong   N_e_abs_pow  = 0;                                // counter for electrons absorbed at the powered electrode
Ullong   N_e_abs_gnd  = 0;                                // counter for electrons absorbed at the grounded electrode
Ullong   N_i_abs_pow  = 0;                                // counter for ions absorbed at the powered electrode
Ullong   N_i_abs_gnd  = 0;                                // counter for ions absorbed at the grounded electrode

// electron energy probability function

int           N_EEPF  = 2000;                             // number of energy bins in Electron Energy Probability Function (EEPF)
double        DE_EEPF = 0.05;                             // resolution of EEPF [eV]
using eepf_vector = vector<double>;                       // array for EEPF
eepf_vector  eepf;                                        // time integrated EEPF in the center of the plasma

// ion flux-energy distributions

int           N_IFED  = 200;                              // number of energy bins in Ion Flux-Energy Distributions (IFEDs)
double        DE_IFED = 1.0;                              // resolution of IFEDs [eV]
using ifed_vector = vector<int>;                          // array for IFEDs
ifed_vector  ifed_pow;                                    // IFED at the powered electrode
ifed_vector  ifed_gnd;                                    // IFED at the grounded electrode
double       mean_i_energy_pow;                           // mean ion energy at the powered electrode
double       mean_i_energy_gnd;                           // mean ion energy at the grounded electrode

// ion impact angle distributions

using iang_vector = vector<int>;                          // array for ion impact angle distributions
iang_vector  iang_pow;                                    // ion impact angles at the powered electrode
iang_vector  iang_gnd;                                    // ion impact angles at the grounded electrode

// spatio-temporal (XT) distributions

using xt_distr = vector<double>;                          // array for XT distributions (decimal numbers)


xt_distr pot_xt;                                          // XT distribution of the potential
xt_distr efield_xt;                                       // XT distribution of the electric field
xt_distr ne_xt;                                           // XT distribution of the electron density
xt_distr ni_xt;                                           // XT distribution of the ion density
xt_distr ue_xt;                                           // XT distribution of the mean electron velocity
xt_distr ui_xt;                                           // XT distribution of the mean ion velocity
xt_distr je_xt;                                           // XT distribution of the electron current density
xt_distr ji_xt;                                           // XT distribution of the ion current density
xt_distr powere_xt;                                       // XT distribution of the electron powering (power absorption) rate
xt_distr poweri_xt;                                       // XT distribution of the ion powering (power absorption) rate
xt_distr meanee_xt;                                       // XT distribution of the mean electron energy
xt_distr meanei_xt;                                       // XT distribution of the mean ion energy
xt_distr counter_e_xt;                                    // XT counter for electron properties
xt_distr counter_i_xt;                                    // XT counter for ion properties
xt_distr ioniz_rate_xt;                                   // XT distribution of the ionisation rate

double    mean_energy_accu_center    = 0;                 // mean electron energy accumulator in the center of the gap
Ullong    mean_energy_counter_center = 0;                 // mean electron energy counter in the center of the gap
Ullong    N_e_coll                   = 0;                 // counter for electron collisions
Ullong    N_i_coll                   = 0;                 // counter for ion collisions

double    Time;                                           // total simulated time (from the beginning of the simulation)
int       cycle,no_of_cycles,cycles_done;                 // current cycle and total cycles in the run, cycles completed (from the beginning of the simulation)

int       arg1;                                           // used for reading command line arguments
bool      measurement_mode;                               // flag that controls measurements and data saving

ofstream  datafile;                                      // stream to external file for saving convergence data (rank 0 only)
const string DATA_DIR = "data";

//------------------------------------------------------------------------//
// C++ Mersenne Twister 19937 generator                                   //
// R01(MTgen) will genarate uniform distribution over [0,1) interval      //
// RMB(MTgen) will generate Maxwell-Boltzmann distribution (of gas atoms) //
//------------------------------------------------------------------------//

std::mt19937 MTgen;
std::uniform_real_distribution<> R01(0.0, 1.0);
std::normal_distribution<> RMB;
std::normal_distribution<> RME;

int mpi_rank = 0;
int mpi_size = 1;
Ullong N_e_total = 0;
Ullong N_i_total = 0;

vector<double> ne_center_time;
vector<double> ne_center_value;

void init_mpi(int &argc, char **&argv){
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
}

string data_path(const string &name){
    return DATA_DIR + "/" + name;
}

size_t local_count_from_total(size_t total){
    size_t base = total / static_cast<size_t>(mpi_size);
    size_t rem = total % static_cast<size_t>(mpi_size);
    return base + ((static_cast<size_t>(mpi_rank) < rem) ? 1 : 0);
}

int read_seed_from_env(){
    const char *env = std::getenv("EDUPIC_SEED");
    if (env == nullptr) { return 5489; }
    try {
        return std::stoi(env);
    } catch (...) {
        return 5489;
    }
}

void seed_rng(){
    int base_seed = read_seed_from_env();
    std::seed_seq seq{base_seed, mpi_rank};
    MTgen.seed(seq);
}

string trim_ws(const string &s){
    const auto start = s.find_first_not_of(" \t\r\n");
    if (start == string::npos) { return ""; }
    const auto end = s.find_last_not_of(" \t\r\n");
    return s.substr(start, end - start + 1);
}

string strip_inline_comment(const string &s){
    auto out = s;
    auto hash_pos = out.find('#');
    auto slash_pos = out.find("//");
    auto cut = string::npos;
    if (hash_pos != string::npos) { cut = hash_pos; }
    if (slash_pos != string::npos) { cut = min(cut, slash_pos); }
    if (cut != string::npos) { out = out.substr(0, cut); }
    return trim_ws(out);
}

bool apply_param_kv(const string &key, const string &value_raw){
    const auto value = strip_inline_comment(value_raw);
    if (value.empty()) { return false; }
    try {
        if (key == "N_G") { N_G = stoi(value); return true; }
        if (key == "N_T") { N_T = stoi(value); return true; }
        if (key == "FREQUENCY") { FREQUENCY = stod(value); return true; }
        if (key == "VOLTAGE") { VOLTAGE = stod(value); return true; }
        if (key == "L") { L = stod(value); return true; }
        if (key == "PRESSURE") { PRESSURE = stod(value); return true; }
        if (key == "TEMPERATURE") { TEMPERATURE = stod(value); return true; }
        if (key == "WEIGHT") { WEIGHT = stod(value); return true; }
        if (key == "ELECTRODE_AREA") { ELECTRODE_AREA = stod(value); return true; }
        if (key == "N_INIT") { N_INIT = stoi(value); return true; }
        if (key == "N_SUB") { N_SUB = stoi(value); return true; }
        if (key == "N_BIN") { N_BIN = stoi(value); return true; }
        if (key == "GAMMA_I") { GAMMA_I = stod(value); return true; }
        if (key == "GAMMA_E") { GAMMA_E = stod(value); return true; }
        if (key == "N_IANG") { N_IANG = stoi(value); return true; }
        if (key == "DE_IANG") { DE_IANG = stod(value); return true; }
        if (key == "USE_BOLTZMANN_ELECTRONS") { USE_BOLTZMANN_ELECTRONS = (stoi(value) != 0); return true; }
        if (key == "NE_CENTER_MODE") { NE_CENTER_MODE = stoi(value); return true; }
        if (key == "NE_CENTER_FIXED") { NE_CENTER_FIXED = stod(value); return true; }
        if (key == "TE_INIT_EEV") { TE_INIT_EEV = stod(value); return true; }
        if (key == "TE_MIN_EEV") { TE_MIN_EEV = stod(value); return true; }
        if (key == "BOLTZMANN_ITER") { BOLTZMANN_ITER = stoi(value); return true; }
        if (key == "BOLTZMANN_DAMP") { BOLTZMANN_DAMP = stod(value); return true; }
        if (key == "NE_CENTER_TABLE") { NE_CENTER_TABLE = value; return true; }
        if (key == "N_EEPF") { N_EEPF = stoi(value); return true; }
        if (key == "DE_EEPF") { DE_EEPF = stod(value); return true; }
        if (key == "N_IFED") { N_IFED = stoi(value); return true; }
        if (key == "DE_IFED") { DE_IFED = stod(value); return true; }
    } catch (...) {
        return false;
    }
    return false;
}

void broadcast_string(string &value){
    int len = static_cast<int>(value.size());
    MPI_Bcast(&len, 1, MPI_INT, 0, MPI_COMM_WORLD);
    value.resize(static_cast<size_t>(len));
    if (len > 0) {
        MPI_Bcast(value.data(), len, MPI_CHAR, 0, MPI_COMM_WORLD);
    }
}

bool load_params_csv(const string &path, bool require){
    ifstream f(path);
    if (!f) {
        if (require) {
            if (mpi_rank == 0) {
                cerr<<">> eduPIC: ERROR: parameter file not found: "<<path<<endl;
            }
            return false;
        }
        return true;
    }

    string line;
    int line_no = 0;
    while (getline(f, line)){
        ++line_no;
        auto s = trim_ws(line);
        if (s.empty() || s[0] == '#') { continue; }
        if (s.size() > 1 && s[0] == '/' && s[1] == '/') { continue; }
        auto sep = s.find(',');
        if (sep == string::npos) { sep = s.find('='); }
        if (sep == string::npos) {
            if (mpi_rank == 0) {
                cerr<<">> eduPIC: Warning: invalid param line "<<line_no<<": "<<line<<endl;
            }
            continue;
        }
        auto key = trim_ws(s.substr(0, sep));
        auto value = trim_ws(s.substr(sep + 1));
        if (!apply_param_kv(key, value)) {
            if (mpi_rank == 0) {
                cerr<<">> eduPIC: Warning: unknown or invalid param at line "<<line_no<<": "<<line<<endl;
            }
        }
    }
    return true;
}

void broadcast_params(){
    MPI_Bcast(&N_G, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&N_T, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&FREQUENCY, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&VOLTAGE, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&L, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&PRESSURE, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&TEMPERATURE, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&WEIGHT, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&ELECTRODE_AREA, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&N_INIT, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&N_SUB, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&N_BIN, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&GAMMA_I, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&GAMMA_E, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&N_IANG, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&DE_IANG, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&N_EEPF, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&DE_EEPF, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&N_IFED, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&DE_IFED, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    {
        int boltz = USE_BOLTZMANN_ELECTRONS ? 1 : 0;
        MPI_Bcast(&boltz, 1, MPI_INT, 0, MPI_COMM_WORLD);
        USE_BOLTZMANN_ELECTRONS = (boltz != 0);
    }
    MPI_Bcast(&NE_CENTER_MODE, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&NE_CENTER_FIXED, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&TE_INIT_EEV, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&TE_MIN_EEV, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&BOLTZMANN_ITER, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&BOLTZMANN_DAMP, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    broadcast_string(NE_CENTER_TABLE);
}

bool validate_params(string &error){
    if (N_G < 2) { error = "N_G must be >= 2"; return false; }
    if (N_T <= 0) { error = "N_T must be > 0"; return false; }
    if (N_BIN <= 0) { error = "N_BIN must be > 0"; return false; }
    if ((N_T % N_BIN) != 0) { error = "N_T must be divisible by N_BIN"; return false; }
    if (N_SUB <= 0) { error = "N_SUB must be > 0"; return false; }
    if (N_EEPF <= 0) { error = "N_EEPF must be > 0"; return false; }
    if (N_IFED <= 0) { error = "N_IFED must be > 0"; return false; }
    if (N_IANG <= 0) { error = "N_IANG must be > 0"; return false; }
    if (DE_IANG <= 0.0) { error = "DE_IANG must be > 0"; return false; }
    if (GAMMA_I < 0.0 || GAMMA_I > 1.0) { error = "GAMMA_I must be in [0,1]"; return false; }
    if (GAMMA_E < 0.0 || GAMMA_E > 1.0) { error = "GAMMA_E must be in [0,1]"; return false; }
    if (FREQUENCY <= 0.0) { error = "FREQUENCY must be > 0"; return false; }
    if (L <= 0.0) { error = "L must be > 0"; return false; }
    if (PRESSURE <= 0.0) { error = "PRESSURE must be > 0"; return false; }
    if (TEMPERATURE <= 0.0) { error = "TEMPERATURE must be > 0"; return false; }
    if (ELECTRODE_AREA <= 0.0) { error = "ELECTRODE_AREA must be > 0"; return false; }
    if (WEIGHT <= 0.0) { error = "WEIGHT must be > 0"; return false; }
    if (USE_BOLTZMANN_ELECTRONS) {
        if (NE_CENTER_MODE < 1 || NE_CENTER_MODE > 2) { error = "NE_CENTER_MODE must be 1 or 2"; return false; }
        if (NE_CENTER_MODE == 1 && NE_CENTER_FIXED <= 0.0) { error = "NE_CENTER_FIXED must be > 0"; return false; }
        if (TE_INIT_EEV <= 0.0) { error = "TE_INIT_EEV must be > 0"; return false; }
        if (TE_MIN_EEV <= 0.0) { error = "TE_MIN_EEV must be > 0"; return false; }
        if (BOLTZMANN_ITER <= 0) { error = "BOLTZMANN_ITER must be > 0"; return false; }
        if (BOLTZMANN_DAMP <= 0.0 || BOLTZMANN_DAMP > 1.0) { error = "BOLTZMANN_DAMP must be in (0,1]"; return false; }
    }
    return true;
}

bool load_ne_center_table(const string &path, bool require){
    ifstream f(path);
    if (!f) {
        if (require && mpi_rank == 0) {
            cerr<<">> eduPIC: ERROR: center density table not found: "<<path<<endl;
        }
        return false;
    }
    vector<pair<double, double>> rows;
    string line;
    int line_no = 0;
    while (getline(f, line)) {
        ++line_no;
        auto s = strip_inline_comment(line);
        if (s.empty()) { continue; }
        auto sep = s.find(',');
        if (sep == string::npos) { sep = s.find('='); }
        if (sep == string::npos) {
            if (mpi_rank == 0) {
                cerr<<">> eduPIC: Warning: invalid ne table line "<<line_no<<": "<<line<<endl;
            }
            continue;
        }
        try {
            auto t = stod(trim_ws(s.substr(0, sep)));
            auto ne0 = stod(trim_ws(s.substr(sep + 1)));
            rows.emplace_back(t, ne0);
        } catch (...) {
            if (mpi_rank == 0) {
                cerr<<">> eduPIC: Warning: invalid ne table line "<<line_no<<": "<<line<<endl;
            }
        }
    }
    if (rows.empty()) {
        if (require && mpi_rank == 0) {
            cerr<<">> eduPIC: ERROR: center density table is empty: "<<path<<endl;
        }
        return false;
    }
    sort(rows.begin(), rows.end(), [](const auto &a, const auto &b){ return a.first < b.first; });
    ne_center_time.clear();
    ne_center_value.clear();
    ne_center_time.reserve(rows.size());
    ne_center_value.reserve(rows.size());
    for (const auto &row : rows) {
        ne_center_time.push_back(row.first);
        ne_center_value.push_back(row.second);
    }
    return true;
}

void compute_derived_params(){
    PERIOD      = 1.0 / FREQUENCY;
    DT_E        = PERIOD / static_cast<double>(N_T);
    DT_I        = static_cast<double>(N_SUB) * DT_E;
    DX          = L / static_cast<double>(N_G - 1);
    INV_DX      = 1.0 / DX;
    GAS_DENSITY = PRESSURE / (K_BOLTZMANN * TEMPERATURE);
    OMEGA       = TWO_PI * FREQUENCY;
    N_XT        = N_T / N_BIN;
}

void init_simulation_arrays(){
    efield.assign(static_cast<size_t>(N_G), 0.0);
    pot.assign(static_cast<size_t>(N_G), 0.0);
    e_density.assign(static_cast<size_t>(N_G), 0.0);
    i_density.assign(static_cast<size_t>(N_G), 0.0);
    cumul_e_density.assign(static_cast<size_t>(N_G), 0.0);
    cumul_i_density.assign(static_cast<size_t>(N_G), 0.0);

    eepf.assign(static_cast<size_t>(N_EEPF), 0.0);
    ifed_pow.assign(static_cast<size_t>(N_IFED), 0);
    ifed_gnd.assign(static_cast<size_t>(N_IFED), 0);
    iang_pow.assign(static_cast<size_t>(N_IANG), 0);
    iang_gnd.assign(static_cast<size_t>(N_IANG), 0);

    const auto xt_size = static_cast<size_t>(N_G) * static_cast<size_t>(N_XT);
    pot_xt.assign(xt_size, 0.0);
    efield_xt.assign(xt_size, 0.0);
    ne_xt.assign(xt_size, 0.0);
    ni_xt.assign(xt_size, 0.0);
    ue_xt.assign(xt_size, 0.0);
    ui_xt.assign(xt_size, 0.0);
    je_xt.assign(xt_size, 0.0);
    ji_xt.assign(xt_size, 0.0);
    powere_xt.assign(xt_size, 0.0);
    poweri_xt.assign(xt_size, 0.0);
    meanee_xt.assign(xt_size, 0.0);
    meanei_xt.assign(xt_size, 0.0);
    counter_e_xt.assign(xt_size, 0.0);
    counter_i_xt.assign(xt_size, 0.0);
    ioniz_rate_xt.assign(xt_size, 0.0);

    mean_energy_accu_center = 0.0;
    mean_energy_counter_center = 0;
    N_e_coll = 0;
    N_i_coll = 0;
    N_e_abs_pow = 0;
    N_e_abs_gnd = 0;
    N_i_abs_pow = 0;
    N_i_abs_gnd = 0;
}

void configure_simulation(const string &path, bool require){
    bool ok = true;
    if (mpi_rank == 0) {
        ok = load_params_csv(path, require);
    }
    int ok_flag = ok ? 1 : 0;
    MPI_Bcast(&ok_flag, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (ok_flag == 0) {
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    broadcast_params();
    string error;
    if (!validate_params(error)) {
        if (mpi_rank == 0) {
            cerr<<">> eduPIC: ERROR: invalid parameters: "<<error<<endl;
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    if (USE_BOLTZMANN_ELECTRONS && NE_CENTER_MODE == 2) {
        if (!load_ne_center_table(NE_CENTER_TABLE, true)) {
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }
    compute_derived_params();
    init_simulation_arrays();
    RMB = std::normal_distribution<>(0.0, sqrt(K_BOLTZMANN * TEMPERATURE / AR_MASS));
    RME = std::normal_distribution<>(0.0, sqrt(K_BOLTZMANN * TEMPERATURE / E_MASS));
}

size_t xt_index(size_t p, size_t t_index){
    return p * static_cast<size_t>(N_XT) + t_index;
}

//----------------------------------------------------------------------------//
//  electron cross sections: A V Phelps & Z Lj Petrovic, PSST 8 R21 (1999)    //
//----------------------------------------------------------------------------//

void set_electron_cross_sections_ar(void){
    cout<<">> eduPIC: Setting e- / Ar cross sections"<<endl;

    auto qmel = [](auto en){ return 1e-20*(fabs(6.0 / pow(1.0 + (en/0.1) + pow(en/0.6,2.0), 3.3)
        - 1.1 * pow(en, 1.4) / (1.0 + pow(en/15.0, 1.2)) / sqrt(1.0 + pow(en/5.5, 2.5) + pow(en/60.0, 4.1)))
        + 0.05 / pow(1.0 + en/10.0, 2.0) + 0.01 * pow(en, 3.0) / (1.0 + pow(en/12.0, 6.0))); };

    auto qexc= [](const auto &en){ if(en>E_EXC_TH){ return 1e-20*(0.034 * pow(en-11.5, 1.1) * (1.0 + pow(en/15.0, 2.8)) / (1.0 + pow(en/23.0, 5.5))
            + 0.023 * (en-11.5) / pow(1.0 + en/80.0, 1.9)); } else { return 0.0;} };

    auto qion = [](const auto &en){ if(en>E_ION_TH){ return 1e-20*(970.0 * (en-15.8) / pow(70.0 + en, 2.0) +
                0.06 * pow(en-15.8, 2.0) * exp(-en/9)); } else {return 0.0;} };

    vector<float> e(CS_RANGES);
    e[0]=DE_CS;             
    generate(e.begin()+1,e.end(),[i=1]()mutable{return DE_CS*(i++); });   // electron energy

    transform(e.begin(),e.end(),sigma[E_ELA].begin(),qmel);    // cross section for e- / Ar elastic collision
    transform(e.begin(),e.end(),sigma[E_EXC].begin(),qexc);    // cross section for e- / Ar excitation
    transform(e.begin(),e.end(),sigma[E_ION].begin(),qion);    // cross section for e- / Ar ionization
}

//-------------------------------------------------------------------//
//  ion cross sections: A. V. Phelps, J. Appl. Phys. 76, 747 (1994)  //
//-------------------------------------------------------------------//

void set_ion_cross_sections_ar(void){
    cout<<">> eduPIC: Setting Ar+ / Ar cross sections"<<endl;
    auto qiso = [](const auto &e_lab){ return 2e-19 * pow(e_lab,-0.5) / (1.0 + e_lab) +
                                    3e-19 * e_lab / pow(1.0 + e_lab / 3.0, 2.0); };

    auto qmom= [](const auto &e_lab){ return 1.15e-18 * pow(e_lab,-0.1) * pow(1.0 + 0.015 / e_lab, 0.6); };

    auto qback = [&](const auto &x){ return (qmom(x)-qiso(x))/2.0; };

    vector<float> e(CS_RANGES);
    e[0]=2.0*DE_CS;
    generate(e.begin()+1,e.end(),[i=1]()mutable{return 2.0*DE_CS*(i++); });   // ion energy in the laboratory frame of reference

    transform(e.begin(),e.end(),sigma[I_ISO].begin(),qiso);     // cross section for Ar+ / Ar isotropic part of elastic scattering
    transform(e.begin(),e.end(),sigma[I_BACK].begin(),qback);   // cross section for Ar+ / Ar backward elastic scattering
}

//----------------------------------------------------------------------//
//  calculation of total cross sections for electrons and ions          //
//----------------------------------------------------------------------//

void calc_total_cross_sections(void){

    for(size_t i{0}; i<CS_RANGES; ++i){
        sigma_tot_e[i] = (sigma[E_ELA][i] + sigma[E_EXC][i] + sigma[E_ION][i]) * GAS_DENSITY;   // total macroscopic cross section of electrons
        sigma_tot_i[i] = (sigma[I_ISO][i] + sigma[I_BACK][i]) * GAS_DENSITY;                    // total macroscopic cross section of ions
    }
}

//----------------------------------------------------------------------//
//  test of cross sections for electrons and ions                       //
//----------------------------------------------------------------------//

void test_cross_sections(void){
    ofstream f("cross_sections.dat");                             // cross sections saved in data file: cross_sections.dat
    ostream_iterator<float> tofile(f, "\n");

    for(size_t i{0}; i<CS_RANGES;++i){f<<i*DE_CS<<endl;}
    for(const auto & v:sigma){
        copy(v.begin(),v.end(),tofile);
    }
    f.close();
}

//---------------------------------------------------------------------//
// find upper limit of collision frequencies                           //
//---------------------------------------------------------------------//

double max_electron_coll_freq (void){
    double e,v,nu,nu_max;
    nu_max = 0;
    for(size_t i{0}; i<CS_RANGES; ++i){
        e  = i * DE_CS;
        v  = sqrt(2.0 * e * EV_TO_J / E_MASS);
        nu = v * sigma_tot_e[i];
        if (nu > nu_max) {nu_max = nu;}
    }
    return nu_max;
}

double max_ion_coll_freq (void){
    double e,g,nu,nu_max;
    nu_max = 0;
    for(size_t i{0}; i<CS_RANGES; ++i){
        e  = i * DE_CS;
        g  = sqrt(2.0 * e * EV_TO_J / MU_ARAR);
        nu = g * sigma_tot_i[i];
        if (nu > nu_max) nu_max = nu;
    }
    return nu_max;
}

//----------------------------------------------------------------------//
// initialization of the simulation by placing a given number of        //
// electrons and ions at random positions between the electrodes        //
//----------------------------------------------------------------------//

void init(int nseed){
    size_t local_n = local_count_from_total(static_cast<size_t>(nseed));
    N_e = local_n;
    N_i = local_n;
    x_e.resize(local_n);
    x_i.resize(local_n);
    vx_e.resize(local_n, 0.0);                                      // initial velocity components of the electron
    vy_e.resize(local_n, 0.0);
    vz_e.resize(local_n, 0.0);
    vx_i.resize(local_n, 0.0);                                      // initial velocity components of the ion
    vy_i.resize(local_n, 0.0);
    vz_i.resize(local_n, 0.0);
    generate(x_e.begin(),x_e.end(),[=](){return L*R01(MTgen);});     // initial random position of the electron
    generate(x_i.begin(),x_i.end(),[=](){return L*R01(MTgen);});     // initial random position of the ion
}

//----------------------------------------------------------------------//
// e / Ar collision  (cold gas approximation)                           //
//----------------------------------------------------------------------//

void collision_electron (double xe, double &vxe, double &vye, double &vze, const int &eindex){
    const double F1 = E_MASS  / (E_MASS + AR_MASS);
    const double F2 = AR_MASS / (E_MASS + AR_MASS);
    double t0,t1,t2,rnd;
    double g,g2,gx,gy,gz,wx,wy,wz,theta,phi;
    double chi,eta,chi2,eta2,sc,cc,se,ce,st,ct,sp,cp,energy,e_sc,e_ej;
    
    // calculate relative velocity before collision & velocity of the centre of mass
    
    gx = vxe;                             
    gy = vye;
    gz = vze;
    g  = sqrt(gx * gx + gy * gy + gz * gz);
    wx = F1 * vxe;
    wy = F1 * vye;
    wz = F1 * vze;
    
    // find Euler angles
    
    if (gx == 0) {theta = 0.5 * PI;}
    else {theta = atan2(sqrt(gy * gy + gz * gz),gx);}
    if (gy == 0) {
        if (gz > 0){phi = 0.5 * PI;} else {phi = - 0.5 * PI;}
    } else {phi = atan2(gz, gy);}
    st  = sin(theta);
    ct  = cos(theta);
    sp  = sin(phi);
    cp  = cos(phi);
    
    // choose the type of collision based on the cross sections
    // take into account energy loss in inelastic collisions
    // generate scattering and azimuth angles
    // in case of ionization handle the 'new' electron
    
    t0   =     sigma[E_ELA][eindex];
    t1   = t0 +sigma[E_EXC][eindex];
    t2   = t1 +sigma[E_ION][eindex];
    rnd  = R01(MTgen);
    if (rnd < (t0/t2)){                              // elastic scattering
        chi = acos(1.0 - 2.0 * R01(MTgen));          // isotropic scattering
        eta = TWO_PI * R01(MTgen);                   // azimuthal angle
    } else if (rnd < (t1/t2)){                       // excitation
        energy = 0.5 * E_MASS * g * g;               // electron energy
        energy = fabs(energy - E_EXC_TH * EV_TO_J);  // subtract energy loss for excitation
        g = sqrt(2.0 * energy / E_MASS);             // relative velocity after energy loss
        chi = acos(1.0 - 2.0 * R01(MTgen));          // isotropic scattering
        eta = TWO_PI * R01(MTgen);                   // azimuthal angle
    } else {                                         // ionization
        energy = 0.5 * E_MASS * g * g;               // electron energy
        energy = fabs(energy - E_ION_TH * EV_TO_J);  // subtract energy loss for ionization
        e_ej = 10.0 * tan(R01(MTgen) * atan(energy/EV_TO_J / 20.0)) * EV_TO_J;   // energy of the emitted electron
        e_sc = fabs(energy - e_ej);                  // energy of incoming electron after collision
        g    = sqrt(2.0 * e_sc / E_MASS);            // relative velocity of incoming (original) electron
        g2   = sqrt(2.0 * e_ej / E_MASS);            // relative velocity of emitted (new) electron
        chi  = acos(sqrt(e_sc / energy));            // scattering angle for incoming electron
        chi2 = acos(sqrt(e_ej / energy));            // scattering angle for emitted electrons
        eta  = TWO_PI * R01(MTgen);                  // azimuthal angle for incoming electron
        eta2 = eta + PI;                             // azimuthal angle for emitted electron
        sc  = sin(chi2);
        cc  = cos(chi2);
        se  = sin(eta2);
        ce  = cos(eta2);
        gx  = g2 * (ct * cc - st * sc * ce);
        gy  = g2 * (st * cp * cc + ct * cp * sc * ce - sp * sc * se);
        gz  = g2 * (st * sp * cc + ct * sp * sc * ce + cp * sc * se);
        N_e++;                                        // add new electron
        x_e.push_back(xe);
        vx_e.push_back(wx + F2 * gx);
        vy_e.push_back(wy + F2 * gy);
        vz_e.push_back(wz + F2 * gz);
        N_i++;                                        // add new ion
        x_i.push_back(xe);
        vx_i.push_back(RMB(MTgen));                   // velocity is sampled from background thermal distribution
        vy_i.push_back(RMB(MTgen));
        vz_i.push_back(RMB(MTgen));
    }
    
    // scatter the primary electron

    sc  = sin(chi);
    cc  = cos(chi);
    se  = sin(eta);
    ce  = cos(eta);
    
    // compute new relative velocity:
    
    gx = g * (ct * cc - st * sc * ce);
    gy = g * (st * cp * cc + ct * cp * sc * ce - sp * sc * se);
    gz = g * (st * sp * cc + ct * sp * sc * ce + cp * sc * se);
    
    // post-collision velocity of the colliding electron
    
    vxe = wx + F2 * gx;
    vye = wy + F2 * gy;
    vze = wz + F2 * gz;
}

//----------------------------------------------------------------------//
// Ar+ / Ar collision                                                   //
//----------------------------------------------------------------------//

void collision_ion (double &vx_1, double &vy_1, double &vz_1,
                    double &vx_2, double &vy_2, double &vz_2, const int &e_index){
    double   g,gx,gy,gz,wx,wy,wz,rnd;
    double   theta,phi,chi,eta,st,ct,sp,cp,sc,cc,se,ce,t1,t2;
    
    // calculate relative velocity before collision
    // random Maxwellian target atom already selected (vx_2,vy_2,vz_2 velocity components of target atom come with the call)
    

    gx = vx_1-vx_2;
    gy = vy_1-vy_2;
    gz = vz_1-vz_2;
    g  = sqrt(gx * gx + gy * gy + gz * gz);
    wx = 0.5 * (vx_1 + vx_2);
    wy = 0.5 * (vy_1 + vy_2);
    wz = 0.5 * (vz_1 + vz_2);

    // find Euler angles:

    if (gx == 0) {theta = 0.5 * PI;} else {theta = atan2(sqrt(gy * gy + gz * gz),gx);}
    if (gy == 0) {
        if (gz > 0){phi = 0.5 * PI;} else {phi = - 0.5 * PI;}
    } else {phi = atan2(gz, gy);}


    // determine the type of collision based on cross sections and generate scattering angle

    t1  =     sigma[I_ISO][e_index];
    t2  = t1 +sigma[I_BACK][e_index];
    rnd = R01(MTgen);
    if  (rnd < (t1 /t2)){                                  // isotropic scattering
        chi = acos(1.0 - 2.0 * R01(MTgen));                // isotropic scattering angle
    } else {                                               // backward scattering
        chi = PI;                                          // backward scattering angle
    }
    eta = TWO_PI * R01(MTgen);                             // azimuthal angle
    sc  = sin(chi);
    cc  = cos(chi);
    se  = sin(eta);
    ce  = cos(eta);
    st  = sin(theta);
    ct  = cos(theta);
    sp  = sin(phi);
    cp  = cos(phi);

    // compute new relative velocity:

    gx = g * (ct * cc - st * sc * ce);
    gy = g * (st * cp * cc + ct * cp * sc * ce - sp * sc * se);
    gz = g * (st * sp * cc + ct * sp * sc * ce + cp * sc * se);

    // post-collision velocity of the ion

    vx_1 = wx + 0.5 * gx;
    vy_1 = wy + 0.5 * gy;
    vz_1 = wz + 0.5 * gz;
}

//----------------------------------------------------------------------
// solve Poisson equation (Thomas algorithm)    
//----------------------------------------------------------------------

double get_center_density(double t){
    if (!USE_BOLTZMANN_ELECTRONS) { return 0.0; }
    if (NE_CENTER_MODE == 1 || ne_center_time.empty()) { return NE_CENTER_FIXED; }
    if (t <= ne_center_time.front()) { return ne_center_value.front(); }
    if (t >= ne_center_time.back()) { return ne_center_value.back(); }
    auto it = upper_bound(ne_center_time.begin(), ne_center_time.end(), t);
    auto idx = static_cast<size_t>(it - ne_center_time.begin());
    if (idx == 0) { return ne_center_value.front(); }
    double t1 = ne_center_time[idx - 1];
    double t2 = ne_center_time[idx];
    double v1 = ne_center_value[idx - 1];
    double v2 = ne_center_value[idx];
    double w = (t - t1) / (t2 - t1);
    return v1 + w * (v2 - v1);
}

double get_electron_temperature_eV(){
    if (!USE_BOLTZMANN_ELECTRONS) { return TE_INIT_EEV; }
    double local_sum = mean_energy_accu_center;
    Ullong local_count = mean_energy_counter_center;
    double global_sum = 0.0;
    Ullong global_count = 0;
    MPI_Allreduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&local_count, &global_count, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
    if (global_count == 0) { return TE_INIT_EEV; }
    double mean_energy = global_sum / static_cast<double>(global_count);
    double Te = (2.0 / 3.0) * mean_energy;
    return max(Te, TE_MIN_EEV);
}

void compute_boltzmann_e_density(double n0, double Te_eV, const xvector &pot_in, xvector &out){
    const size_t center = static_cast<size_t>(N_G / 2);
    const double phi0 = pot_in[center];
    const double inv_Te = 1.0 / Te_eV;
    const double max_arg = 50.0;
    for (size_t i = 0; i < static_cast<size_t>(N_G); ++i) {
        double arg = (pot_in[i] - phi0) * inv_Te;
        if (arg > max_arg) { arg = max_arg; }
        if (arg < -max_arg) { arg = -max_arg; }
        out[i] = n0 * exp(arg);
    }
}

void compute_efield_from_potential(const xvector &rho1){
    const double S = 1.0 / (2.0 * DX);
    size_t i;
    for (i = 1; i <= static_cast<size_t>(N_G - 2); ++i) {
        efield[i] = (pot[i - 1] - pot[i + 1]) * S;
    }
    efield.front() = (pot[0]     - pot[1])     * INV_DX - rho1.front() * DX / (2.0 * EPSILON0);
    efield.back()  = (pot[N_G-2] - pot[N_G-1]) * INV_DX + rho1.back()  * DX / (2.0 * EPSILON0);
}

void solve_Poisson (const xvector &rho1, const double &tt){
    const double A =  1.0;
    const double B = -2.0;
    const double C =  1.0;
    const double ALPHA = -DX * DX / EPSILON0;
    xvector g(static_cast<size_t>(N_G), 0.0), w(static_cast<size_t>(N_G), 0.0), f(static_cast<size_t>(N_G), 0.0);
    size_t  i;

    // apply potential to the electrodes - boundary conditions

    pot.front() = VOLTAGE * cos(OMEGA * tt);    // potential at the powered electrode
    pot.back()  = 0.0;                          // potential at the grounded electrode

    // solve Poisson equation

    for(i=1; i<=N_G-2; ++i) f[i] = ALPHA * rho1[i];
    f[1] -= pot.front();
    f[N_G-2] -= pot.back();
    w[1] = C/B;
    g[1] = f[1]/B;
    for(i=2; i<=N_G-2; ++i){
        w[i] = C / (B - A * w[i-1]);
        g[i] = (f[i] - A * g[i-1]) / (B - A * w[i-1]);
    }
    pot[N_G-2] = g[N_G-2];
    for (i=N_G-3; i>0; --i) pot[i] = g[i] - w[i] * pot[i+1];    // potential at the grid points between the electrodes

    // compute electric field

    compute_efield_from_potential(rho1);
}

void solve_Boltzmann_Poisson(const xvector &i_density_in, const double &tt){
    const double n0 = get_center_density(tt);
    const double Te = get_electron_temperature_eV();
    xvector rho(static_cast<size_t>(N_G), 0.0);
    xvector pot_old(static_cast<size_t>(N_G), 0.0);
    for (int iter = 0; iter < BOLTZMANN_ITER; ++iter) {
        compute_boltzmann_e_density(n0, Te, pot, e_density);
        transform(i_density_in.begin(), i_density_in.end(), e_density.begin(), rho.begin(), [](auto x, auto y){ return E_CHARGE * (x - y); });
        pot_old = pot;
        solve_Poisson(rho, tt);
        if (BOLTZMANN_DAMP < 1.0) {
            for (size_t i = 0; i < static_cast<size_t>(N_G); ++i) {
                pot[i] = pot_old[i] + BOLTZMANN_DAMP * (pot[i] - pot_old[i]);
            }
        }
    }
    compute_boltzmann_e_density(n0, Te, pot, e_density);
    transform(i_density_in.begin(), i_density_in.end(), e_density.begin(), rho.begin(), [](auto x, auto y){ return E_CHARGE * (x - y); });
    compute_efield_from_potential(rho);
}

//---------------------------------------------------------------------//
// simulation of one radiofrequency cycle                              //
//---------------------------------------------------------------------//

void do_one_cycle (void){
    const double DV       = ELECTRODE_AREA * DX;
    const double FACTOR_W = WEIGHT / DV;
    const double FACTOR_E = DT_E / E_MASS * E_CHARGE;
    const double FACTOR_I = DT_I / AR_MASS * E_CHARGE;
    const double MIN_X = 0.45 * L;                       // min. position for EEPF collection
    const double MAX_X = 0.55 * L;                       // max. position for EEPF collection
    size_t      k, t, p, energy_index;
    double   rmod, rint, g, g_sqr, gx, gy, gz, vx_a, vy_a, vz_a, e_x, energy, nu, p_coll, v_sqr, velocity;
    double   mean_v, rate;
    bool     out;
    xvector  rho(static_cast<size_t>(N_G), 0.0);
    size_t      t_index;

    for (t=0; t<N_T; t++){         // a RF period is divided into N_T equal time intervals (time step DT_E)
        Time += DT_E;              // update of the total simulated time
        t_index = t / N_BIN;       // index for XT distributions

        // step 1: compute densities at grid points

        if (!USE_BOLTZMANN_ELECTRONS) {
            fill(e_density.begin(),e_density.end(),0.0);             // electron density - computed in every time step
            for(k=0; k<N_e; ++k){
                rmod = modf(x_e[k] * INV_DX, &rint);
                p    = static_cast<size_t>(rint);
                e_density[p]   += (1.0-rmod) * FACTOR_W;             
                e_density[p+1] += rmod * FACTOR_W;
            }
            MPI_Allreduce(MPI_IN_PLACE, e_density.data(), static_cast<int>(N_G), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            e_density.front() *= 2.0;
            e_density.back()  *= 2.0;
            transform(cumul_e_density.begin(),cumul_e_density.end(),e_density.begin(),cumul_e_density.begin(),[](auto x, auto y){return x+y;});
        } else {
            fill(e_density.begin(), e_density.end(), 0.0);
        }

        if ((t % N_SUB) == 0) {                                  // ion density - computed in every N_SUB-th time steps (subcycling)
            fill(i_density.begin(),i_density.end(),0.0);
            for(k=0; k<N_i; ++k){
                rmod = modf(x_i[k] * INV_DX, &rint);
                p    = static_cast<size_t>(rint);
                i_density[p]   += (1.0-rmod) * FACTOR_W;         
                i_density[p+1] += rmod * FACTOR_W;
            }
            MPI_Allreduce(MPI_IN_PLACE, i_density.data(), static_cast<int>(N_G), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            i_density.front() *= 2.0;
            i_density.back()  *= 2.0;
        }
        transform(cumul_i_density.begin(),cumul_i_density.end(),i_density.begin(),cumul_i_density.begin(),[](auto x, auto y){return x+y;});

        // step 2: solve Poisson equation
        
        if (USE_BOLTZMANN_ELECTRONS) {
            solve_Boltzmann_Poisson(i_density, Time);                         // compute potential and electric field
            transform(cumul_e_density.begin(),cumul_e_density.end(),e_density.begin(),cumul_e_density.begin(),[](auto x, auto y){return x+y;});
        } else {
            // get charge density
            transform(i_density.begin(),i_density.end(),e_density.begin(),rho.begin(),[](auto x, auto y){return E_CHARGE*(x-y);});
            solve_Poisson(rho,Time);                                           // compute potential and electric field
        }

        // steps 3 & 4: move particles according to electric field interpolated to particle positions

        for(k=0; k<N_e; k++){                      // move all electrons in every time step
            rmod = modf(x_e[k] * INV_DX, &rint);
            p    = static_cast<size_t>(rint);
            e_x  = (1.0-rmod)*efield[p] + rmod*efield[p+1];

            if (measurement_mode) {
               
                // measurements: 'x' and 'v' are needed at the same time, i.e. old 'x' and mean 'v'
                
                mean_v = vx_e[k] - 0.5 * e_x * FACTOR_E;
                counter_e_xt[xt_index(p, t_index)]   += (1.0-rmod);
                counter_e_xt[xt_index(p + 1, t_index)] += rmod;
                ue_xt[xt_index(p, t_index)]   += (1.0-rmod) * mean_v;
                ue_xt[xt_index(p + 1, t_index)] += rmod * mean_v;
                v_sqr  = mean_v * mean_v + vy_e[k] * vy_e[k] + vz_e[k] * vz_e[k];
                energy = 0.5 * E_MASS * v_sqr / EV_TO_J;
                meanee_xt[xt_index(p, t_index)]   += (1.0-rmod) * energy;
                meanee_xt[xt_index(p + 1, t_index)] += rmod * energy;
                energy_index = min( static_cast<int>(energy / DE_CS + 0.5), CS_RANGES-1);
                velocity = sqrt(v_sqr);
                rate = sigma[E_ION][energy_index] * velocity * DT_E * GAS_DENSITY;
                ioniz_rate_xt[xt_index(p, t_index)]   += (1.0-rmod) * rate;
                ioniz_rate_xt[xt_index(p + 1, t_index)] += rmod * rate;

                // measure EEPF in the center
                
                if ((MIN_X < x_e[k]) && (x_e[k] < MAX_X)){
                    energy_index = static_cast<int>(energy / DE_EEPF);
                    if (energy_index < N_EEPF) {eepf[energy_index] += 1.0;}
                    mean_energy_accu_center += energy;
                    mean_energy_counter_center++;
                }
            }

            // update velocity and position

            vx_e[k] -= e_x * FACTOR_E;
            x_e[k]  += vx_e[k] * DT_E;
        }

        if ((t % N_SUB) == 0) {                    // move all ions in every N_SUB-th time steps (subcycling)
            for(k=0; k<N_i; k++){
                rmod = modf(x_i[k] * INV_DX, &rint);
                p    = static_cast<size_t>(rint);
                e_x  = (1.0-rmod)*efield[p] + rmod*efield[p+1];

                if (measurement_mode) {
                    
                    // measurements: 'x' and 'v' are needed at the same time, i.e. old 'x' and mean 'v'

                    mean_v = vx_i[k] + 0.5 * e_x * FACTOR_I;
                    counter_i_xt[xt_index(p, t_index)]   += (1.0-rmod);
                    counter_i_xt[xt_index(p + 1, t_index)] += rmod;
                    ui_xt[xt_index(p, t_index)]   += (1.0-rmod) * mean_v;
                    ui_xt[xt_index(p + 1, t_index)] += rmod * mean_v;
                    v_sqr  = mean_v * mean_v + vy_i[k] * vy_i[k] + vz_i[k] * vz_i[k];
                    energy = 0.5 * AR_MASS * v_sqr / EV_TO_J;
                    meanei_xt[xt_index(p, t_index)]   += (1.0-rmod) * energy;
                    meanei_xt[xt_index(p + 1, t_index)] += rmod * energy;
                }
                
                // update velocity and position
                
                vx_i[k] += e_x * FACTOR_I;
                x_i[k]  += vx_i[k] * DT_I;
            }
        }
        

        // step 5: check boundaries
        k = 0;
        while(k < x_e.size()) {    // check boundaries for all electrons in every time step
            out = false;
            if (x_e[k] < 0) {                                   // the electron hits the powered electrode
                if (R01(MTgen) < GAMMA_E) {
                    x_e[k] = 0.0;
                    vx_e[k] = fabs(vx_e[k]);
                } else {
                    N_e_abs_pow++; out = true;
                }
            } else if (x_e[k] > L) {                            // the electron hits the grounded electrode
                if (R01(MTgen) < GAMMA_E) {
                    x_e[k] = L;
                    vx_e[k] = -fabs(vx_e[k]);
                } else {
                    N_e_abs_gnd++; out = true;
                }
            }
            if (out) {                                        // remove the electron, if out
                x_e[k]=x_e.back(); x_e.pop_back();
                vx_e[k]=vx_e.back(); vx_e.pop_back();
                vy_e[k]=vy_e.back(); vy_e.pop_back();
                vz_e[k]=vz_e.back(); vz_e.pop_back();
                N_e--;
            } else k++;
        }

        if ((t % N_SUB) == 0) {           // check boundaries for all ions in every N_SUB-th time steps (subcycling)
            k = 0;
            while(k < x_i.size()) {
                out = false;
                if (x_i[k] < 0) {             // the ion is out at the powered electrode
                    N_i_abs_pow++;
                    out    = true;
                    v_sqr  = vx_i[k] * vx_i[k] + vy_i[k] * vy_i[k] + vz_i[k] * vz_i[k];
                    energy = 0.5 * AR_MASS * v_sqr / EV_TO_J;
                    energy_index = static_cast<int>(energy / DE_IFED);
                    if (energy_index < N_IFED) {ifed_pow[energy_index]++;}   // save IFED at the powered electrode
                    if (measurement_mode) {
                        const double v_sqr_xy = vx_i[k] * vx_i[k] + vy_i[k] * vy_i[k];
                        if (v_sqr_xy > 0.0) {
                            const double velocity_xy = sqrt(v_sqr_xy);
                            const double cos_theta = min(1.0, fabs(vx_i[k]) / velocity_xy);
                            const double theta_deg = acos(cos_theta) * 180.0 / PI;
                            const int ang_index = min(static_cast<int>(theta_deg / DE_IANG), N_IANG - 1);
                            if (ang_index >= 0) {iang_pow[static_cast<size_t>(ang_index)]++;}
                        }
                    }
                    if (R01(MTgen) < GAMMA_I) {
                        N_e++;
                        x_e.push_back(0.0);
                        vx_e.push_back(fabs(RME(MTgen)));
                        vy_e.push_back(RME(MTgen));
                        vz_e.push_back(RME(MTgen));
                    }
                }
                if (x_i[k] > L) {             // the ion is out at the grounded electrode
                    N_i_abs_gnd++;
                    out    = true;
                    v_sqr  = vx_i[k] * vx_i[k] + vy_i[k] * vy_i[k] + vz_i[k] * vz_i[k];
                    energy = 0.5 * AR_MASS * v_sqr / EV_TO_J;
                    energy_index = static_cast<int>(energy / DE_IFED);
                    if (energy_index < N_IFED) {ifed_gnd[energy_index]++;}   // save IFED at the grounded electrode
                    if (measurement_mode) {
                        const double v_sqr_xy = vx_i[k] * vx_i[k] + vy_i[k] * vy_i[k];
                        if (v_sqr_xy > 0.0) {
                            const double velocity_xy = sqrt(v_sqr_xy);
                            const double cos_theta = min(1.0, fabs(vx_i[k]) / velocity_xy);
                            const double theta_deg = acos(cos_theta) * 180.0 / PI;
                            const int ang_index = min(static_cast<int>(theta_deg / DE_IANG), N_IANG - 1);
                            if (ang_index >= 0) {iang_gnd[static_cast<size_t>(ang_index)]++;}
                        }
                    }
                    if (R01(MTgen) < GAMMA_I) {
                        N_e++;
                        x_e.push_back(L);
                        vx_e.push_back(-fabs(RME(MTgen)));
                        vy_e.push_back(RME(MTgen));
                        vz_e.push_back(RME(MTgen));
                    }
                }
                if (out) {                    // delete the ion, if out
                        x_i[k]=x_i.back(); x_i.pop_back();
                        vx_i[k]=vx_i.back(); vx_i.pop_back();
                        vy_i[k]=vy_i.back(); vy_i.pop_back();
                        vz_i[k]=vz_i.back(); vz_i.pop_back();
                        N_i--;
                } else k++;
            }
        }

        // step 6: collisions

        for (k=0; k<N_e; ++k){                         // checking for occurrence of a collision for all electrons in every time step
            v_sqr = vx_e[k] * vx_e[k] + vy_e[k] * vy_e[k] + vz_e[k] * vz_e[k];
            velocity = sqrt(v_sqr);
            energy   = 0.5 * E_MASS * v_sqr / EV_TO_J;
            energy_index = min(static_cast<int>(energy / DE_CS + 0.5), CS_RANGES-1);
            nu = sigma_tot_e[energy_index] * velocity;
            p_coll = 1 - exp(- nu * DT_E);             // collision probability for electrons
            if (R01(MTgen) < p_coll) {                 // electron collision takes place
                collision_electron(x_e[k], vx_e[k], vy_e[k], vz_e[k], energy_index);
                N_e_coll++;
            }
        }

        if ((t % N_SUB) == 0) {                        // checking for occurrence of a collision for all ions in every N_SUB-th time steps (subcycling)
            for (k=0; k<N_i; ++k){
                vx_a = RMB(MTgen);                     // pick velocity components of a random gas atoms
                vy_a = RMB(MTgen);
                vz_a = RMB(MTgen);
                gx   = vx_i[k] - vx_a;                  // compute the relative velocity of the collision partners
                gy   = vy_i[k] - vy_a;
                gz   = vz_i[k] - vz_a;
                g_sqr = gx * gx + gy * gy + gz * gz;
                g = sqrt(g_sqr);
                energy = 0.5 * MU_ARAR * g_sqr / EV_TO_J;
                energy_index = min( static_cast<int>(energy / DE_CS + 0.5), CS_RANGES-1);    
                nu = sigma_tot_i[energy_index] * g;
                p_coll = 1 - exp(- nu * DT_I);         // collision probability for ions
                if (R01(MTgen)< p_coll) {              // ion collision takes place
                    collision_ion (vx_i[k], vy_i[k], vz_i[k], vx_a, vy_a, vz_a, energy_index);
                    N_i_coll++;
                }
            }
        }

        if (measurement_mode && mpi_rank == 0) {

            // collect data from the grid:

            for (p=0; p<N_G; p++) {
                pot_xt   [xt_index(p, t_index)] += pot[p];
                efield_xt[xt_index(p, t_index)] += efield[p];
                ne_xt    [xt_index(p, t_index)] += e_density[p];
                ni_xt    [xt_index(p, t_index)] += i_density[p];
            }
        }

           if ((t % 1000) == 0) {
               Ullong local_e_step = static_cast<Ullong>(N_e);
               Ullong local_i_step = static_cast<Ullong>(N_i);
               MPI_Allreduce(&local_e_step, &N_e_total, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
               MPI_Allreduce(&local_i_step, &N_i_total, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
               if (mpi_rank == 0) {
                   cout<<" c = "<<setw(8)<<cycle<<"  t = "<<setw(8)<<t<<"  #e = "<<setw(8)<<N_e_total<<"  #i = "<<setw(8)<<N_i_total<<endl;
               }
           }
    }
    Ullong local_e = static_cast<Ullong>(N_e);
    Ullong local_i = static_cast<Ullong>(N_i);
    MPI_Allreduce(&local_e, &N_e_total, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&local_i, &N_i_total, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
    if (mpi_rank == 0) { datafile<<cycle<<" "<<N_e_total<<" "<<N_i_total<<endl; }
}

void reduce_array_double(double *data, int count){
    if (mpi_rank == 0) {
        MPI_Reduce(MPI_IN_PLACE, data, count, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    } else {
        MPI_Reduce(data, nullptr, count, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    }
}

void reduce_array_int(int *data, int count){
    if (mpi_rank == 0) {
        MPI_Reduce(MPI_IN_PLACE, data, count, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    } else {
        MPI_Reduce(data, nullptr, count, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    }
}

void reduce_scalar_double(double &value){
    if (mpi_rank == 0) {
        MPI_Reduce(MPI_IN_PLACE, &value, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    } else {
        MPI_Reduce(&value, nullptr, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    }
}

void reduce_scalar_ullong(Ullong &value){
    if (mpi_rank == 0) {
        MPI_Reduce(MPI_IN_PLACE, &value, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    } else {
        MPI_Reduce(&value, nullptr, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    }
}

void reduce_measurements_to_root(void){
    Ullong local_e = static_cast<Ullong>(N_e);
    Ullong local_i = static_cast<Ullong>(N_i);
    if (mpi_rank == 0) {
        MPI_Reduce(&local_e, &N_e_total, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(&local_i, &N_i_total, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    } else {
        MPI_Reduce(&local_e, nullptr, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(&local_i, nullptr, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    }

    if (!measurement_mode) { return; }

    reduce_scalar_double(mean_energy_accu_center);
    reduce_scalar_ullong(mean_energy_counter_center);
    reduce_scalar_ullong(N_e_coll);
    reduce_scalar_ullong(N_i_coll);
    reduce_scalar_ullong(N_e_abs_pow);
    reduce_scalar_ullong(N_e_abs_gnd);
    reduce_scalar_ullong(N_i_abs_pow);
    reduce_scalar_ullong(N_i_abs_gnd);

    reduce_array_double(eepf.data(), N_EEPF);
    reduce_array_int(ifed_pow.data(), N_IFED);
    reduce_array_int(ifed_gnd.data(), N_IFED);
    reduce_array_int(iang_pow.data(), N_IANG);
    reduce_array_int(iang_gnd.data(), N_IANG);
    reduce_array_double(counter_e_xt.data(), N_G * N_XT);
    reduce_array_double(counter_i_xt.data(), N_G * N_XT);
    reduce_array_double(ue_xt.data(), N_G * N_XT);
    reduce_array_double(ui_xt.data(), N_G * N_XT);
    reduce_array_double(meanee_xt.data(), N_G * N_XT);
    reduce_array_double(meanei_xt.data(), N_G * N_XT);
    reduce_array_double(ioniz_rate_xt.data(), N_G * N_XT);
}

//---------------------------------------------------------------------//
// save and load particle coordinates                                  //
//---------------------------------------------------------------------//

void save_particle_data(){
    int local_e = static_cast<int>(N_e);
    int local_i = static_cast<int>(N_i);
    vector<int> counts_e, counts_i, displs_e, displs_i;

    if (mpi_rank == 0) {
        counts_e.resize(mpi_size);
        counts_i.resize(mpi_size);
    }

    MPI_Gather(&local_e, 1, MPI_INT, counts_e.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Gather(&local_i, 1, MPI_INT, counts_i.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

    vector<double> x_e_all, vx_e_all, vy_e_all, vz_e_all;
    vector<double> x_i_all, vx_i_all, vy_i_all, vz_i_all;

    int total_e = 0;
    int total_i = 0;
    if (mpi_rank == 0) {
        displs_e.resize(mpi_size);
        displs_i.resize(mpi_size);
        displs_e[0] = 0;
        displs_i[0] = 0;
        for (int r = 0; r < mpi_size; ++r) {
            total_e += counts_e[r];
            total_i += counts_i[r];
            if (r > 0) {
                displs_e[r] = displs_e[r-1] + counts_e[r-1];
                displs_i[r] = displs_i[r-1] + counts_i[r-1];
            }
        }
        x_e_all.resize(total_e);
        vx_e_all.resize(total_e);
        vy_e_all.resize(total_e);
        vz_e_all.resize(total_e);
        x_i_all.resize(total_i);
        vx_i_all.resize(total_i);
        vy_i_all.resize(total_i);
        vz_i_all.resize(total_i);
    }

    MPI_Gatherv(x_e.data(), local_e, MPI_DOUBLE, x_e_all.data(), counts_e.data(), displs_e.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gatherv(vx_e.data(), local_e, MPI_DOUBLE, vx_e_all.data(), counts_e.data(), displs_e.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gatherv(vy_e.data(), local_e, MPI_DOUBLE, vy_e_all.data(), counts_e.data(), displs_e.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gatherv(vz_e.data(), local_e, MPI_DOUBLE, vz_e_all.data(), counts_e.data(), displs_e.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    MPI_Gatherv(x_i.data(), local_i, MPI_DOUBLE, x_i_all.data(), counts_i.data(), displs_i.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gatherv(vx_i.data(), local_i, MPI_DOUBLE, vx_i_all.data(), counts_i.data(), displs_i.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gatherv(vy_i.data(), local_i, MPI_DOUBLE, vy_i_all.data(), counts_i.data(), displs_i.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gatherv(vz_i.data(), local_i, MPI_DOUBLE, vz_i_all.data(), counts_i.data(), displs_i.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (mpi_rank == 0) {
        ofstream f(data_path("picdata.bin"),ios::binary);
        f.write(reinterpret_cast<char*>(&Time),sizeof(double));
        f.write(reinterpret_cast<char*>(&cycles_done),sizeof(int));
        f.write(reinterpret_cast<char*>(&total_e),sizeof(int));
        f.write(reinterpret_cast<char*>(x_e_all.data()),total_e*sizeof(double));
        f.write(reinterpret_cast<char*>(vx_e_all.data()),total_e*sizeof(double));
        f.write(reinterpret_cast<char*>(vy_e_all.data()),total_e*sizeof(double));
        f.write(reinterpret_cast<char*>(vz_e_all.data()),total_e*sizeof(double));
        f.write(reinterpret_cast<char*>(&total_i),sizeof(int));
        f.write(reinterpret_cast<char*>(x_i_all.data()),total_i*sizeof(double));
        f.write(reinterpret_cast<char*>(vx_i_all.data()),total_i*sizeof(double));
        f.write(reinterpret_cast<char*>(vy_i_all.data()),total_i*sizeof(double));
        f.write(reinterpret_cast<char*>(vz_i_all.data()),total_i*sizeof(double));
        f.close();

        cout<<">> eduPIC: data saved : "<<total_e<<" electrons "<<total_i<<" ions, "
        <<cycles_done<<" cycles completed, time is "<<scientific<<Time<<" [s]"<<endl;
    }
}

//---------------------------------------------------------------------//
// load particle coordinates                                           //
//---------------------------------------------------------------------//

void load_particle_data(){
    int total_e = 0;
    int total_i = 0;
    vector<double> x_e_all, vx_e_all, vy_e_all, vz_e_all;
    vector<double> x_i_all, vx_i_all, vy_i_all, vz_i_all;

    if (mpi_rank == 0) {
        ifstream f(data_path("picdata.bin"),std::ios::binary);
        if (f.fail()) {
            cout<<">> eduPIC: ERROR: No particle data file found, try running initial cycle using argument '0'"<<endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        f.read(reinterpret_cast<char*>(&Time),sizeof(double));
        f.read(reinterpret_cast<char*>(&cycles_done),sizeof(int));
        f.read(reinterpret_cast<char*>(&total_e),sizeof(int));
        x_e_all.resize(total_e);
        vx_e_all.resize(total_e);
        vy_e_all.resize(total_e);
        vz_e_all.resize(total_e);
        f.read(reinterpret_cast<char*>(x_e_all.data()),total_e*sizeof(double));
        f.read(reinterpret_cast<char*>(vx_e_all.data()),total_e*sizeof(double));
        f.read(reinterpret_cast<char*>(vy_e_all.data()),total_e*sizeof(double));
        f.read(reinterpret_cast<char*>(vz_e_all.data()),total_e*sizeof(double));
        f.read(reinterpret_cast<char*>(&total_i),sizeof(int));
        x_i_all.resize(total_i);
        vx_i_all.resize(total_i);
        vy_i_all.resize(total_i);
        vz_i_all.resize(total_i);
        f.read(reinterpret_cast<char*>(x_i_all.data()),total_i*sizeof(double));
        f.read(reinterpret_cast<char*>(vx_i_all.data()),total_i*sizeof(double));
        f.read(reinterpret_cast<char*>(vy_i_all.data()),total_i*sizeof(double));
        f.read(reinterpret_cast<char*>(vz_i_all.data()),total_i*sizeof(double));
        f.close();
    }

    MPI_Bcast(&Time, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&cycles_done, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&total_e, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&total_i, 1, MPI_INT, 0, MPI_COMM_WORLD);

    vector<int> counts_e(mpi_size), counts_i(mpi_size), displs_e(mpi_size), displs_i(mpi_size);
    int base_e = total_e / mpi_size;
    int rem_e = total_e % mpi_size;
    int base_i = total_i / mpi_size;
    int rem_i = total_i % mpi_size;
    for (int r = 0; r < mpi_size; ++r) {
        counts_e[r] = base_e + ((r < rem_e) ? 1 : 0);
        counts_i[r] = base_i + ((r < rem_i) ? 1 : 0);
        displs_e[r] = (r == 0) ? 0 : (displs_e[r-1] + counts_e[r-1]);
        displs_i[r] = (r == 0) ? 0 : (displs_i[r-1] + counts_i[r-1]);
    }

    int local_e = counts_e[mpi_rank];
    int local_i = counts_i[mpi_rank];
    N_e = static_cast<size_t>(local_e);
    N_i = static_cast<size_t>(local_i);
    x_e.resize(local_e);
    vx_e.resize(local_e);
    vy_e.resize(local_e);
    vz_e.resize(local_e);
    x_i.resize(local_i);
    vx_i.resize(local_i);
    vy_i.resize(local_i);
    vz_i.resize(local_i);

    MPI_Scatterv(x_e_all.data(), counts_e.data(), displs_e.data(), MPI_DOUBLE, x_e.data(), local_e, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatterv(vx_e_all.data(), counts_e.data(), displs_e.data(), MPI_DOUBLE, vx_e.data(), local_e, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatterv(vy_e_all.data(), counts_e.data(), displs_e.data(), MPI_DOUBLE, vy_e.data(), local_e, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatterv(vz_e_all.data(), counts_e.data(), displs_e.data(), MPI_DOUBLE, vz_e.data(), local_e, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    MPI_Scatterv(x_i_all.data(), counts_i.data(), displs_i.data(), MPI_DOUBLE, x_i.data(), local_i, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatterv(vx_i_all.data(), counts_i.data(), displs_i.data(), MPI_DOUBLE, vx_i.data(), local_i, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatterv(vy_i_all.data(), counts_i.data(), displs_i.data(), MPI_DOUBLE, vy_i.data(), local_i, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatterv(vz_i_all.data(), counts_i.data(), displs_i.data(), MPI_DOUBLE, vz_i.data(), local_i, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (mpi_rank == 0) {
        cout<<">> eduPIC: data loaded : "<<total_e<<" electrons "<<total_i<<" ions, "
        <<cycles_done<<" cycles completed before, time is "<<scientific<<Time<<" [s]"<<endl;
    }
}

//---------------------------------------------------------------------//
// save density data                                                   //
//---------------------------------------------------------------------//

void save_density(void){
    ofstream f(data_path("density.dat"));
    f<<setprecision(12)<<fixed<<scientific;

    auto c=1.0/static_cast<double>(no_of_cycles)/static_cast<double>(N_T);
    for(size_t i{0}; i<static_cast<size_t>(N_G);++i){
        f<<i*DX<<" "<<cumul_e_density[i]*c<<" "<<cumul_i_density[i]*c<<endl;
    }
    f.close();
}

//---------------------------------------------------------------------//
// save EEPF data                                                      //
//---------------------------------------------------------------------//

void save_eepf(void) {
    ofstream f(data_path("eepf.dat"));
    auto h=accumulate(eepf.begin(),eepf.end(),0.0);
    h *= DE_EEPF;
    f<<scientific;
    double energy {};
    for(size_t i{0}; i<static_cast<size_t>(N_EEPF);++i){
        energy=(i + 0.5) * DE_EEPF;
        f<<energy<<" "<<eepf[i] / h / sqrt(energy)<<endl;
    }
    f.close();
}

//---------------------------------------------------------------------//
// save IFED data                                                      //
//---------------------------------------------------------------------//

void save_ifed(void) {
    double p, g, energy;
    ofstream f(data_path("ifed.dat"));
    f<<scientific;
    double h_pow = accumulate(ifed_pow.begin(),ifed_pow.end(),0.0);
    double h_gnd = accumulate(ifed_gnd.begin(),ifed_gnd.end(),0.0);
    h_pow *= DE_IFED;
    h_gnd *= DE_IFED;
    mean_i_energy_pow = 0.0;
    mean_i_energy_gnd = 0.0;
    for(size_t i{0}; i<static_cast<size_t>(N_IFED);++i){
        energy = (i + 0.5) * DE_IFED;
        p = static_cast<double>(ifed_pow[i]) / h_pow;
        g = static_cast<double>(ifed_gnd[i]) / h_gnd;
        f<<energy<<" "<<p<<" "<<g<<endl;
        mean_i_energy_pow += energy * p;
        mean_i_energy_gnd += energy * g;
    }
    f.close();
}

//---------------------------------------------------------------------//
// save ion impact angle distribution data                             //
//---------------------------------------------------------------------//

void save_iang(void) {
    double p, g, angle;
    ofstream f(data_path("iang.dat"));
    f<<scientific;
    double h_pow = accumulate(iang_pow.begin(),iang_pow.end(),0.0);
    double h_gnd = accumulate(iang_gnd.begin(),iang_gnd.end(),0.0);
    h_pow *= DE_IANG;
    h_gnd *= DE_IANG;
    for(size_t i{0}; i<static_cast<size_t>(N_IANG);++i){
        angle = (i + 0.5) * DE_IANG;
        p = (h_pow > 0.0) ? static_cast<double>(iang_pow[i]) / h_pow : 0.0;
        g = (h_gnd > 0.0) ? static_cast<double>(iang_gnd[i]) / h_gnd : 0.0;
        f<<angle<<" "<<p<<" "<<g<<endl;
    }
    f.close();
}

//--------------------------------------------------------------------//
// save XT data                                                       //
//--------------------------------------------------------------------//

void save_xt_1(xt_distr &distr, string fname) {
    ofstream f(fname);
    ostream_iterator<double> tof(f," ");
    auto it=distr.begin();

    f<<setprecision(8)<<fixed<<scientific;
    for(size_t i{0};i<static_cast<size_t>(N_G);++i){
        copy_n(it, N_XT, tof);
        advance(it, N_XT);
        f<<endl;
    }
    f.close();
}



void norm_all_xt(void){    
    // normalize all XT data
    
    double f1 = static_cast<double>(N_XT) / static_cast<double>(no_of_cycles * N_T);
    double f2 = WEIGHT / (ELECTRODE_AREA * DX) / (no_of_cycles * (PERIOD / static_cast<double>(N_XT)));
    
    transform(pot_xt.begin(),pot_xt.end(),pot_xt.begin(),[=](auto y){return f1*y;});
    transform(efield_xt.begin(),efield_xt.end(),efield_xt.begin(),[=](auto y){return f1*y;});
    transform(ne_xt.begin(),ne_xt.end(),ne_xt.begin(),[=](auto y){return f1*y;});
    transform(ni_xt.begin(),ni_xt.end(),ni_xt.begin(),[=](auto y){return f1*y;});

    transform(ue_xt.begin(),ue_xt.end(),counter_e_xt.begin(),ue_xt.begin(),[](auto x, auto y){if(y>0){return x/y;}else{return 0.0;}});
    transform(ue_xt.begin(),ue_xt.end(),ne_xt.begin(),je_xt.begin(),[=](auto x, auto y){return -x*y*E_CHARGE;});
    transform(meanee_xt.begin(),meanee_xt.end(),counter_e_xt.begin(),meanee_xt.begin(),[](auto x, auto y){if(y>0){return x/y;}else{return 0.0;}});
    transform(ioniz_rate_xt.begin(),ioniz_rate_xt.end(),counter_e_xt.begin(),ioniz_rate_xt.begin(),[=](auto x, auto y){if(y>0){return x*f2;}else{return 0.0;}});

    transform(ui_xt.begin(),ui_xt.end(),counter_i_xt.begin(),ui_xt.begin(),[](auto x, auto y){if(y>0){return x/y;}else{return 0.0;}});
    transform(ui_xt.begin(),ui_xt.end(),ni_xt.begin(),ji_xt.begin(),[=](auto x, auto y){return x*y*E_CHARGE;});
    transform(meanei_xt.begin(),meanei_xt.end(),counter_i_xt.begin(),meanei_xt.begin(),[](auto x, auto y){if(y>0){return x/y;}else{return 0.0;}});
 
    transform(je_xt.begin(),je_xt.end(),efield_xt.begin(),powere_xt.begin(),[=](auto x, auto y){return x*y;});
    transform(ji_xt.begin(),ji_xt.end(),efield_xt.begin(),poweri_xt.begin(),[=](auto x, auto y){return x*y;});
}

    
void save_all_xt(void){
    
    save_xt_1(pot_xt, data_path("pot_xt.dat"));
    save_xt_1(efield_xt, data_path("efield_xt.dat"));
    save_xt_1(ne_xt, data_path("ne_xt.dat"));
    save_xt_1(ni_xt, data_path("ni_xt.dat"));
    save_xt_1(je_xt, data_path("je_xt.dat"));
    save_xt_1(ji_xt, data_path("ji_xt.dat"));
    save_xt_1(powere_xt, data_path("powere_xt.dat"));
    save_xt_1(poweri_xt, data_path("poweri_xt.dat"));
    save_xt_1(meanee_xt, data_path("meanee_xt.dat"));
    save_xt_1(meanei_xt, data_path("meanei_xt.dat"));
    save_xt_1(ioniz_rate_xt, data_path("ioniz_xt.dat"));
}

//---------------------------------------------------------------------//
// simulation report including stability and accuracy conditions       //
//---------------------------------------------------------------------//

void check_and_save_info(void){
    ofstream f(data_path("info.txt"));
    string line (80,'-');
    f<<setprecision(4)<<fixed<<scientific;

    double density = cumul_e_density[N_G / 2]
                        / static_cast<double>(no_of_cycles) / static_cast<double>(N_T);            // e density @ center
    double plas_freq = E_CHARGE * sqrt(density / EPSILON0 / E_MASS);                               // e plasma frequency @ center
    double meane = mean_energy_accu_center / static_cast<double>(mean_energy_counter_center);      // e mean energy @ center
    double kT = 2.0 * meane * EV_TO_J / 3.0;                                                       // k T_e @ center (approximate)
    double debye_length = sqrt(EPSILON0 * kT / density) / E_CHARGE;                                // e Debye length @ center
    double sim_time =  static_cast<double>(no_of_cycles) / FREQUENCY;                              // simulated time
    double ecoll_freq = 0.0;
    double icoll_freq = 0.0;
    if (N_e_total > 0) { ecoll_freq = static_cast<double>(N_e_coll) / sim_time / static_cast<double>(N_e_total); }
    if (N_i_total > 0) { icoll_freq = static_cast<double>(N_i_coll) / sim_time / static_cast<double>(N_i_total); }

    f<<"########################## eduPIC simulation report ############################"<<endl;
    f<<"Simulation parameters:"<<endl;
    f<<"Gap distance                          = "<<L<<" [m]"<<endl;
    f<<"# of grid divisions                   = "<<N_G<<endl;
    f<<"Frequency                             = "<<FREQUENCY<<" [Hz]"<<endl;
    f<<"# of time steps / period              = "<<N_T<<endl;
    f<<"# of electron / ion time steps        = "<<N_SUB<<endl;
    f<<"Voltage amplitude                     = "<<VOLTAGE<<" [V]"<<endl;
    f<<"Pressure (Ar)                         = "<<PRESSURE<<" [Pa]"<<endl;
    f<<"Temperature                           = "<<TEMPERATURE<<" [K]"<<endl;
    f<<"Superparticle weight                  = "<<WEIGHT<<endl;
    f<<"# of simulation cycles in this run    = "<<no_of_cycles<<endl;
    f<<line<<endl;
    f<<"Plasma characteristics:"<<endl;  
    f<<"Electron density @ center             = "<<density<<" [m^{-3}]"<<endl;
    f<<"Plasma frequency @ center             = "<<plas_freq<<" [rad/s]"<<endl;
    f<<"Debye length @ center                 = "<<debye_length<<" [m]"<<endl;
    f<<"Electron collision frequency          = "<<ecoll_freq<<" [1/s]"<<endl;
    f<<"Ion collision frequency               = "<<icoll_freq<<" [1/s]"<<endl;
    f<<line<<endl;
    f<<"Stability and accuracy conditions:"<<endl;  
    auto conditions_OK = true;
    auto c = plas_freq * DT_E;
    f<<"Plasma frequency @ center * DT_e      = "<<c<<" (OK if less than 0.20)"<<endl;
    if (c > 0.2) {conditions_OK = false;}
    c = DX / debye_length;
    f<<"DX / Debye length @ center            = "<<c<<" (OK if less than 1.00)"<<endl;
    if (c > 1.0) {conditions_OK = false;}
    c = max_electron_coll_freq() * DT_E;   
    f<<"Max. electron coll. frequency * DT_E  = "<<c<<" (OK if less than 0.05"<<endl;
    if (c > 0.05) {conditions_OK = false;}
    c = max_ion_coll_freq() * DT_I;
    f<<"Max. ion coll. frequency * DT_I       = "<<c<<" (OK if less than 0.05)"<<endl;
    if (c > 0.05) {conditions_OK = false;}
    if (conditions_OK == false){
        f<<line<<endl;
        f<<"** STABILITY AND ACCURACY CONDITION(S) VIOLATED - REFINE SIMULATION SETTINGS! **"<<endl;
        f<<line<<endl;
        f.close();
        f<<">> eduPIC: ERROR: STABILITY AND ACCURACY CONDITION(S) VIOLATED! "<<endl;
        f<<">> eduPIC: for details see 'data/info.txt' and refine simulation settings!"<<endl;
    }
    else{
        // calculate maximum energy for which the Courant condition holds:
        double v_max = DX / DT_E;
        double e_max = 0.5 * E_MASS * v_max * v_max / EV_TO_J;
        f<<"Max e- energy for CFL     condition   = "<<e_max<<endl;
        f<<"Check EEPF to ensure that CFL is fulfilled for the majority of the electrons!"<<endl;
        f<<line<<endl;

        // saving of the following data is done here as some of the further lines need data
        // that are computed / normalized in these functions

        cout<<">> eduPIC: saving diagnostics data"<<endl;
        save_density();
        save_eepf();
        save_ifed();
        save_iang();
        norm_all_xt();
        save_all_xt();
        f<<"Particle characteristics at the electrodes:"<<endl;
        f<<"Ion flux at powered electrode         = "<<N_i_abs_pow * WEIGHT / ELECTRODE_AREA / (no_of_cycles * PERIOD)<<" [m^{-2} s^{-1}]"<<endl;
        f<<"Ion flux at grounded electrode        = "<<N_i_abs_gnd * WEIGHT / ELECTRODE_AREA / (no_of_cycles * PERIOD)<<" [m^{-2} s^{-1}]"<<endl;
        f<<"Mean ion energy at powered electrode  = "<<mean_i_energy_pow<<" [eV]"<<endl;
        f<<"Mean ion energy at grounded electrode = "<<mean_i_energy_gnd<<" [eV]"<<endl;
        f<<"Electron flux at powered electrode    = "<<N_e_abs_pow * WEIGHT / ELECTRODE_AREA / (no_of_cycles * PERIOD)<<" [m^{-2} s^{-1}]"<<endl;
        f<<"Electron flux at grounded electrode   = "<<N_e_abs_gnd * WEIGHT / ELECTRODE_AREA / (no_of_cycles * PERIOD)<<" [m^{-2} s^{-1}]"<<endl;
        
        // calculate spatially and temporally averaged power absorption by the electrons and ions
        

        auto power_e = accumulate(powere_xt.begin(), powere_xt.end(), 0.0) / static_cast<double>(N_XT * N_G);
        auto power_i = accumulate(poweri_xt.begin(), poweri_xt.end(), 0.0) / static_cast<double>(N_XT * N_G);
        f<<line<<endl;
        f<<"Absorbed power calculated as <j*E>:"<<endl;
        f<<"Electron power density (average)      = "<<power_e<<" [W m^{-3}]"<<endl;
        f<<"Ion power density (average)           = "<<power_i<<" [W m^{-3}]"<<endl;
        f<<"Total power density (average)         = "<<power_e+power_i<<" [W m^{-3}]"<<endl;
        f<<line<<endl;
        f.close();  

    }
}

//------------------------------------------------------------------------------------------//
// main                                                                                     //
// command line arguments:                                                                  //
// [1]: number of cycles (0 for init)                                                       //
// [2]: "m" turns on data collection and saving                                             //
//------------------------------------------------------------------------------------------//

int main (int argc, char *argv[]){
    init_mpi(argc, argv);
    const auto start_time = std::chrono::steady_clock::now();
    seed_rng();

    if (mpi_rank == 0) {
        std::filesystem::create_directories(DATA_DIR);
        datafile.open(data_path("conv.dat"), ios_base::app);
    }

    if (mpi_rank == 0) {
        cout<<">> eduPIC: starting..."<<endl;
        cout<<">> eduPIC: **************************************************************************"<<endl;
        cout<<">> eduPIC: Copyright (C) 2021 Z. Donko et al."<<endl;
        cout<<">> eduPIC: This program comes with ABSOLUTELY NO WARRANTY"<<endl;
        cout<<">> eduPIC: This is free software, you are welcome to use, modify and redistribute it"<<endl;
        cout<<">> eduPIC: according to the GNU General Public License, https://www.gnu.org/licenses/"<<endl;
        cout<<">> eduPIC: **************************************************************************"<<endl;
    }

    if (argc == 1) {
        if (mpi_rank == 0) {
            cout<<">> eduPIC: error = need starting_cycle argument"<<endl;
            cout<<">> eduPIC: usage: ./eduPIC <cycles> [m] [params.csv]"<<endl;
        }
        MPI_Finalize();
        return 1;
    } else {
        vector<string> argList(argv+1,argv+argc);
        arg1 = stoi(argList[0]);
        measurement_mode = false;
        if (argc > 2 && argList[1]=="m"){
            measurement_mode = true;                            // measurements will be done
        }
        string params_path = "params.csv";
        if (measurement_mode) {
            if (argList.size() > 2) { params_path = argList[2]; }
        } else {
            if (argList.size() > 1) { params_path = argList[1]; }
        }
        if (mpi_rank == 0) {
            cout<<">> eduPIC: loading parameters from "<<params_path<<endl;
        }
        configure_simulation(params_path, true);
    }
    if (measurement_mode && mpi_rank == 0) {
        cout<<">> eduPIC: measurement mode: on"<<endl;
    } else if (mpi_rank == 0) {
        cout<<">> eduPIC: measurement mode: off"<<endl;
    }
    if (mpi_rank == 0) {
        set_electron_cross_sections_ar();
        set_ion_cross_sections_ar();
    }
    MPI_Bcast(sigma[0].data(), N_CS * CS_RANGES, MPI_FLOAT, 0, MPI_COMM_WORLD);
    calc_total_cross_sections();
    //test_cross_sections(); return 1;

    if (arg1 == 0) {
        ifstream file(data_path("picdata.bin"),std::ios::binary);
        if (file.good()) { file.close();
            if (mpi_rank == 0) {
                cout<<">> eduPIC: Warning: Data from previous calculation are detected."<<endl;
                cout<<"           To start a new simulation from the beginning, please delete all output files before running ./eduPIC 0"<<endl;
                cout<<"           To continue the existing calculation, please specify the number of cycles to run, e.g. ./eduPIC 100"<<endl; 
            }
            MPI_Finalize();
            exit(0); 
        }
        no_of_cycles = 1;                                 
        cycle = 1;                                        // init cycle
        init(N_INIT);                                     // seed initial electrons & ions
        if (mpi_rank == 0) { cout<<">> eduPIC: running initializing cycle"<<endl; }
        Time = 0;
        do_one_cycle();
        cycles_done = 1;
    } else {
        no_of_cycles = arg1;                              // run number of cycles specified in command line
        load_particle_data();                             // read previous configuration from file
        if (mpi_rank == 0) { cout<<">> eduPIC: running "<<no_of_cycles<<" cycle(s)"<<endl; }
        for (cycle=cycles_done+1;cycle<=cycles_done+no_of_cycles;cycle++) {do_one_cycle();}
        cycles_done += no_of_cycles;
    }
    reduce_measurements_to_root();
    if (mpi_rank == 0) { datafile.close(); }
    save_particle_data();
    if (measurement_mode && mpi_rank == 0) {
        check_and_save_info();
    }
    if (mpi_rank == 0) {
        cout<<">> eduPIC: simulation of "<<no_of_cycles<<" cycle(s) is completed."<<endl;
        const auto end_time = std::chrono::steady_clock::now();
        const auto elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(end_time - start_time).count();
        cout<<">> eduPIC: elapsed time = "<<fixed<<setprecision(3)<<elapsed<<" s"<<endl;
    }
    MPI_Finalize();
}
