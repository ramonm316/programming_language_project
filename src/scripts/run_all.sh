#!/bin/bash

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Global variables to track state
DATA_LOADED=0
CURRENT_IMPL=""
RESULTS_FILE="results.txt"
SELECTED_FILE=""

# Clear results file on start
> "$RESULTS_FILE"

# Function to validate numeric input
validate_number() {
    if [[ ! "$1" =~ ^[0-9]+$ ]]; then
        echo -e "${RED}Error: Invalid input. Please enter a number.${NC}"
        return 1
    fi
    return 0
}

# Function to get current timestamp
timestamp() {
    date '+%Y-%m-%d %H:%M:%S'
}

# Function to scan data folder and populate global array
declare -a DATASETS
scan_data_folder() {
    echo -e "\n${BLUE}Available datasets in data folder:${NC}"
    local i=1
    DATASETS=()

    if [ -d "../data" ]; then
        for file in ../data/*.csv; do
            if [ -f "$file" ]; then
                DATASETS+=("$(basename "$file")")
                echo "  $i. $(basename "$file")"
                ((i++))
            fi
        done
    fi

    if [ ${#DATASETS[@]} -eq 0 ]; then
        echo -e "${RED}No CSV files found in data folder.${NC}"
        return 1
    fi

    return 0
}

# Function to load data
load_data() {
    echo -e "\n${YELLOW}Loading and cleaning input data set:${NC}"
    echo "************************************"

    # Get available datasets
    if ! scan_data_folder; then
        return 1
    fi

    if [ ${#DATASETS[@]} -eq 0 ]; then
        return 1
    fi

    echo ""
    read -p "Enter dataset number: " choice

    if ! validate_number "$choice"; then
        return 1
    fi

    if [ "$choice" -lt 1 ] || [ "$choice" -gt ${#DATASETS[@]} ]; then
        echo -e "${RED}Error: Invalid dataset number.${NC}"
        return 1
    fi

    SELECTED_FILE="${DATASETS[$((choice-1))]}"
    echo -e "\n${GREEN}[$(timestamp)] Starting Script${NC}"
    echo -e "${GREEN}[$(timestamp)] Loading training data set: $SELECTED_FILES${NC}"

    # Get file info
    if [ -f "../data/$SELECTED_FILE" ]; then
        local cols=$(head -1 "../data/$SELECTED_FILE" | tr ',' '\n' | wc -l)
        local rows=$(($(wc -l < "../data/$SELECTED_FILE") - 1))

        echo -e "${GREEN}[$(timestamp)] Total Columns Read: $cols${NC}"
        echo -e "${GREEN}[$(timestamp)] Total Rows Read: $rows${NC}"

        DATA_LOADED=1
        echo -e "\n${GREEN}Data loaded successfully!${NC}"
    else
        echo -e "${RED}Error: File not found.${NC}"
        DATA_LOADED = 0
        return 1
    fi
}

# Function to run Linear Regression
run_linear_regression() {
    if [ $DATA_LOADED -eq 0 ]; then
        echo -e "${RED}Error: Please load data first (option 1).${NC}"
        return 1
    fi

    echo -e "\n${YELLOW}Linear Regression (closed-form):${NC}"
    echo "********************************"

    read -p "Target variable [hours-per-week]: " target
    target=${target:-hours-per-week}

    read -p "L2 regularization [0]: " l2
    l2=${l2:-0}

    echo -e "\n${BLUE}Outputs:${NC}"
    echo "*******"

    local start_time=$(date +%s.%N)

    local DATA_PATH="../data/$SELECTED_FILE"

    case $CURRENT_IMPL in
        "C")
            cd ../proc
            if [ ! -f "program" ]; then
                echo "Compiling C++ implementation..."
                make clean && make -j4 > /dev/null 2>&1
            fi
            ./program --train "$DATA_PATH" \
                     --test "$DATA_PATH" \
                     --target "$target" \
                     --algo linear \
                     --l2 "$l2" \
                     --normalize 2>&1 | tee /tmp/output.txt
            cd ../scripts
            ;;
        "Java")
            cd ../oop-java
            if [ ! -f "app/Main.class" ]; then
                echo "Compiling Java implementation..."
                javac $(find . -name "*.java")
            fi
            echo "2" | java app.Main --train "$DATA_PATH" \
                                    --normalize \
                                    --target "$target" \
                                    --l2 "$l2" 2>&1 | tee /tmp/output.txt
            cd ../scripts
            ;;
        "Lisp")
            cd ../fp
            sbcl --script main.lisp --algo linear --train "$DATA_PATH" --l2 "$l2" 2>&1 | tee /tmp/output.txt
            cd ../scripts
            ;;
    esac

    local end_time=$(date +%s.%N)
    local elapsed=$(echo "$end_time - $start_time" | bc)

    # Extract metrics from output
    local rmse=$(grep -i "RMSE" /tmp/output.txt | tail -1 | awk '{print $NF}')
    local r2=$(grep -i "R\^2\|R²" /tmp/output.txt | tail -1 | awk '{print $NF}')
    local sloc=$(grep -i "SLOC" /tmp/output.txt | tail -1 | awk '{print $NF}')

    # Log results
    echo "$CURRENT_IMPL,Linear Regression,$elapsed,$rmse,$r2,$sloc" >> "$RESULTS_FILE"
}

# Function to run Logistic Regression
run_logistic_regression() {
    if [ $DATA_LOADED -eq 0 ]; then
        echo -e "${RED}Error: Please load data first (option 1).${NC}"
        return 1
    fi

    echo -e "\n${YELLOW}Logistic Regression (binary):${NC}"
    echo "*****************************"

    read -p "Target variable [income]: " target
    target=${target:-income}

    read -p "Learning rate [0.2]: " lr
    lr=${lr:-0.2}

    read -p "Epochs [400]: " epochs
    epochs=${epochs:-400}

    read -p "L2 regularization [0.003]: " l2
    l2=${l2:-0.003}

    read -p "Random seed [7]: " seed
    seed=${seed:-7}

    echo -e "\n${BLUE}Outputs:${NC}"
    echo "*******"

    local start_time=$(date +%s.%N)

    local DATA_PATH="../data/$SELECTED_FILE"

    case $CURRENT_IMPL in
        "C")
            cd ../proc
            if [ ! -f "program" ]; then
                echo "Compiling C++ implementation..."
                make clean && make -j4 > /dev/null 2>&1
            fi
            ./program --train "$DATA_PATH" \
                     --test "$DATA_PATH" \
                     --target "$target" \
                     --algo logistic \
                     --lr "$lr" \
                     --epochs "$epochs" \
                     --l2 "$l2" \
                     --normalize 2>&1 | tee /tmp/output.txt
            cd ../scripts
            ;;
        "Java")
            cd ../oop-java
            if [ ! -f "app/Main.class" ]; then
                echo "Compiling Java implementation..."
                javac $(find . -name "*.java") > /dev/null 2>&1
            fi
            printf "1\n3\n" | stdbuf -oL java app.Main --train "$DATA_PATH" \
                                    --normalize \
                                    --lr "$lr" \
                                    --epochs "$epochs" \
                                    --l2 "$l2" \
                                    --seed "$seed" 2>&1 | tee /tmp/output.txt
            cd ../scripts
            ;;
        "Lisp")
            cd ../fp
            sbcl --script main.lisp --algo logistic \
                 --train "$DATA_PATH" \
                 --lr "$lr" --epochs "$epochs" --l2 "$l2" 2>&1 | tee /tmp/output.txt
            cd ../scripts
            ;;
    esac

    local end_time=$(date +%s.%N)
    local elapsed=$(echo "$end_time - $start_time" | bc)

    # Extract metrics
    local acc=$(grep -i "Accuracy" /tmp/output.txt | tail -1 | awk '{print $NF}')
    local f1=$(grep -i "Macro-F1\|F1" /tmp/output.txt | tail -1 | awk '{print $NF}')
    local sloc=$(grep -i "SLOC" /tmp/output.txt | tail -1 | awk '{print $NF}')

    # Log results
    echo "$CURRENT_IMPL,Logistic Regression,$elapsed,$acc,$f1,$sloc" >> "$RESULTS_FILE"
}

# Function to run k-Nearest Neighbors
run_knn() {
    if [ $DATA_LOADED -eq 0 ]; then
        echo -e "${RED}Error: Please load data first (option 1).${NC}"
        return 1
    fi

    echo -e "\n${YELLOW}k-Nearest Neighbors:${NC}"
    echo "********************"

    read -p "Number of neighbors (k) [5]: " k
    k=${k:-5}

    echo -e "\n${BLUE}Outputs:${NC}"
    echo "*******"

    local start_time=$(date +%s.%N)
    local DATA_PATH="../data/$SELECTED_FILE"

    case $CURRENT_IMPL in
        "C")
            cd ../proc
            if [ ! -f "program" ]; then
                make clean && make -j4 > /dev/null 2>&1
            fi
            ./program --train "$DATA_PATH" \
                     --test "$DATA_PATH" \
                     --target income \
                     --algo knn \
                     --k "$k" \
                     --normalize 2>&1 | tee /tmp/output.txt
            cd ../scripts
            ;;
        "Java")
            cd ../oop-java
            if [ ! -f "app/Main.class" ]; then
                javac $(find . -name "*.java") > /dev/null 2>&1
            fi
            echo "4" | java app.Main --train "$DATA_PATH" \
                                    --normalize \
                                    --k "$k" 2>&1 | tee /tmp/output.txt
            cd ../scripts
            ;;
        "Lisp")
            cd ../fp
            sbcl --script main.lisp --algo knn --train "$DATA_PATH" --k "$k" 2>&1 | tee /tmp/output.txt
            cd ../scripts
            ;;
    esac

    local end_time=$(date +%s.%N)
    local elapsed=$(echo "$end_time - $start_time" | bc)

    local acc=$(grep -i "Accuracy" /tmp/output.txt | tail -1 | awk '{print $NF}')
    local f1=$(grep -i "Macro-F1" /tmp/output.txt | tail -1 | awk '{print $NF}')
    local sloc=$(grep -i "SLOC" /tmp/output.txt | tail -1 | awk '{print $NF}')

    echo "$CURRENT_IMPL,k-Nearest Neighbors,$elapsed,$acc,$f1,$sloc" >> "$RESULTS_FILE"
}

# Function to run Decision Tree
run_decision_tree() {
    if [ $DATA_LOADED -eq 0 ]; then
        echo -e "${RED}Error: Please load data first (option 1).${NC}"
        return 1
    fi

    echo -e "\n${YELLOW}Decision Tree (ID3):${NC}"
    echo "********************"

    read -p "Max depth [5]: " depth
    depth=${depth:-5}

    read -p "Number of bins [10]: " bins
    bins=${bins:-10}

    echo -e "\n${BLUE}Outputs:${NC}"
    echo "*******"

    local start_time=$(date +%s.%N)
    local DATA_PATH="../data/$SELECTED_FILE"

    case $CURRENT_IMPL in
        "C")
            cd ../proc
            if [ ! -f "program" ]; then
                make clean && make -j4 > /dev/null 2>&1
            fi
            ./program --train "$DATA_PATH" \
                     --test "$DATA_PATH" \
                     --target income \
                     --algo tree \
                     --max_depth "$depth" \
                     --normalize 2>&1 | tee /tmp/output.txt
            cd ../scripts
            ;;
        "Java")
            cd ../oop-java
            if [ ! -f "app/Main.class" ]; then
                javac $(find . -name "*.java") > /dev/null 2>&1
            fi
            echo "5" | java app.Main --train "$DATA_PATH" \
                                    --normalize \
                                    --max_depth "$depth" \
                                    --bins "$bins" 2>&1 | tee /tmp/output.txt
            cd ../scripts
            ;;
        "Lisp")
            cd ../fp
            sbcl --script main.lisp --algo tree \
                 --train "$DATA_PATH" \
                 --max_depth "$depth" --n_bins "$bins" 2>&1 | tee /tmp/output.txt
            cd ../scripts
            ;;
    esac

    local end_time=$(date +%s.%N)
    local elapsed=$(echo "$end_time - $start_time" | bc)

    local acc=$(grep -i "Accuracy" /tmp/output.txt | tail -1 | awk '{print $NF}')
    local f1=$(grep -i "Macro-F1" /tmp/output.txt | tail -1 | awk '{print $NF}')
    local sloc=$(grep -i "SLOC" /tmp/output.txt | tail -1 | awk '{print $NF}')

    echo "$CURRENT_IMPL,Decision Tree,$elapsed,$acc,$f1,$sloc" >> "$RESULTS_FILE"
}

# Function to run Gaussian Naive Bayes
run_naive_bayes() {
    if [ $DATA_LOADED -eq 0 ]; then
        echo -e "${RED}Error: Please load data first (option 1).${NC}"
        return 1
    fi

    echo -e "\n${YELLOW}Gaussian Naive Bayes:${NC}"
    echo "*********************"

    read -p "Variance smoothing [1e-9]: " smooth
    smooth=${smooth:-1e-9}

    echo -e "\n${BLUE}Outputs:${NC}"
    echo "*******"

    local start_time=$(date +%s.%N)
    local DATA_PATH="../data/$SELECTED_FILE"

    case $CURRENT_IMPL in
        "C")
            cd ../proc
            if [ ! -f "program" ]; then
                make clean && make -j4 > /dev/null 2>&1
            fi
            ./program --train "$DATA_PATH" \
                     --test "$DATA_PATH" \
                     --target income \
                     --algo nb \
                     --normalize 2>&1 | tee /tmp/output.txt
            cd ../scripts
            ;;
        "Java")
            cd ../oop-java
            if [ ! -f "app/Main.class" ]; then
                javac $(find . -name "*.java") > /dev/null 2>&1
            fi
            echo "6" | java app.Main --train "$DATA_PATH" \
                                    --normalize \
                                    --smoothing "$smooth" 2>&1 | tee /tmp/output.txt
            cd ../scripts
            ;;
        "Lisp")
            cd ../fp
            sbcl --script main.lisp --algo nb --train "$DATA_PATH" 2>&1 | tee /tmp/output.txt
            cd ../scripts
            ;;
    esac

    local end_time=$(date +%s.%N)
    local elapsed=$(echo "$end_time - $start_time" | bc)

    local acc=$(grep -i "Accuracy" /tmp/output.txt | tail -1 | awk '{print $NF}')
    local f1=$(grep -i "Macro-F1" /tmp/output.txt | tail -1 | awk '{print $NF}')
    local sloc=$(grep -i "SLOC" /tmp/output.txt | tail -1 | awk '{print $NF}')

    echo "$CURRENT_IMPL,Gaussian Naive Bayes,$elapsed,$acc,$f1,$sloc" >> "$RESULTS_FILE"
}

# Function to print implementation results
print_impl_results() {
    echo -e "\n${YELLOW}$CURRENT_IMPL Results:${NC}"
    echo "******************************"

    printf "%-8s %-25s %-15s %-16s %-16s %-8s\n" \
           "Impl" "Algorithm" "TrainTime" "TestMetric1" "TestMetric2" "SLOC"
    echo "-----------------------------------------------------------------------------------------"

    if [ -f "$RESULTS_FILE" ]; then
        while IFS=',' read -r impl algo time m1 m2 sloc; do
            if [ "$impl" == "$CURRENT_IMPL" ]; then
                printf "%-8s %-25s %-15s %-16s %-16s %-8s\n" \
                       "$impl" "$algo" "${time}s" "$m1" "$m2" "$sloc"
            fi
        done < "$RESULTS_FILE"
    else
        echo "No results available yet."
    fi
}

# Function to print general comparison results
print_general_results() {
    echo -e "\n${YELLOW}General Results (Comparison):${NC}"
    echo "*****************************"

    printf "%-8s %-25s %-15s %-16s %-16s %-8s\n" \
           "Impl" "Algorithm" "TrainTime" "TestMetric1" "TestMetric2" "SLOC"
    echo "-----------------------------------------------------------------------------------------"

    if [ -f "$RESULTS_FILE" ] && [ -s "$RESULTS_FILE" ]; then
        while IFS=',' read -r impl algo time m1 m2 sloc; do
            printf "%-8s %-25s %-15s %-16s %-16s %-8s\n" \
                   "$impl" "$algo" "${time}s" "$m1" "$m2" "$sloc"
        done < "$RESULTS_FILE"
    else
        echo "No results available yet. Please run some algorithms first."
    fi
}

# Algorithm menu
algorithm_menu() {
    while true; do
        echo -e "\n${BLUE}******************************************************${NC}"
        echo -e "${GREEN}You have selected $CURRENT_IMPL${NC}"
        echo -e "${BLUE}******************************************************${NC}"
        echo "Please select an option:"
        echo "(1) Load data"
        echo "(2) Linear Regression (closed-form)"
        echo "(3) Logistic Regression (binary)"
        echo "(4) k-Nearest Neighbors"
        echo "(5) Decision Tree (ID3)"
        echo "(6) Gaussian Naive Bayes"
        echo "(7) Print results"
        echo "(8) Back to main menu"
        echo ""

        read -p "Enter option: " choice

        if ! validate_number "$choice"; then
            continue
        fi

        case $choice in
            1) load_data ;;
            2) run_linear_regression ;;
            3) run_logistic_regression ;;
            4) run_knn ;;
            5) run_decision_tree ;;
            6) run_naive_bayes ;;
            7) print_impl_results ;;
            8) DATA_LOADED=0; break ;;
            *) echo -e "${RED}Invalid option. Please try again.${NC}" ;;
        esac
    done
}

# Main menu
main_menu() {
    while true; do
        echo -e "\n${BLUE}******************************************************${NC}"
        echo -e "${GREEN}Welcome to the AI/ML Library Implementation Comparison${NC}"
        echo -e "${BLUE}******************************************************${NC}"
        echo "Please select an implementation to run:"
        echo "(1) Procedural (C/C++)"
        echo "(2) Object-Oriented (Java)"
        echo "(3) Functional (Lisp)"
        echo "(4) Print General Results"
        echo "(5) Quit"
        echo ""

        read -p "Enter option: " choice

        if ! validate_number "$choice"; then
            continue
        fi

        case $choice in
            1)
                CURRENT_IMPL="C"
                algorithm_menu
                ;;
            2)
                CURRENT_IMPL="Java"
                algorithm_menu
                ;;
            3)
                CURRENT_IMPL="Lisp"
                algorithm_menu
                ;;
            4)
                print_general_results
                ;;
            5)
                echo -e "\n${GREEN}Thank you for using the comparison tool!${NC}"
                exit 0
                ;;
            *)
                echo -e "${RED}Invalid option. Please enter a number between 1 and 5.${NC}"
                ;;
        esac
    done
}

# Check for required commands
check_dependencies() {
    local missing=0

    if ! command -v bc &> /dev/null; then
        echo -e "${RED}Error: 'bc' is required but not installed.${NC}"
        missing=1
    fi

    if ! command -v sbcl &> /dev/null && [ "$1" == "3" ]; then
        echo -e "${YELLOW}Warning: 'sbcl' not found. Lisp implementation will not work.${NC}"
    fi

    if ! command -v javac &> /dev/null && [ "$1" == "2" ]; then
        echo -e "${YELLOW}Warning: 'javac' not found. Java implementation will not work.${NC}"
    fi

    if ! command -v g++ &> /dev/null && [ "$1" == "1" ]; then
        echo -e "${YELLOW}Warning: 'g++' not found. C++ implementation will not work.${NC}"
    fi

    return $missing
}

# Start the script
check_dependencies
main_menu
