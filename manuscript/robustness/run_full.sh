#!/usr/bin/env bash
# Run every analysis step for each group. Progress goes to logs/status.txt and
# step output to logs/<group>_<step>.log. A failed step skips the rest of its group.
# With no arguments all four groups run. To run one group, pass its folder name
# and cell selection options, e.g. run_full.sh microglia --cell-type Microglia.
# Run detached:
#   setsid nohup manuscript/robustness/run_full.sh > /dev/null 2>&1 < /dev/null &
cd "$(dirname "$0")/../.."
ROOT=manuscript/robustness
mkdir -p $ROOT/logs
export PYTHONPATH=src
PY="env/bin/python -u manuscript"
status() { date "+$1 %F %T" >> $ROOT/logs/status.txt; }

run_group() {
	local name=$1 dir=$ROOT/$1; shift
	local sel="$*"
	mkdir -p $dir
	rm -f $dir/alzheimers_*
	status "$name start"
	local steps=(
		"analysis|$PY/alzheimers_analysis.py --out-dir $dir --n-seeds 10 $sel"
		"null|$PY/alzheimers_coexpression_null.py --out-dir $dir --n-shuffles 25 $sel"
		"de_plots|$PY/alzheimers_de_plots.py --out-dir $dir"
		"rf_baseline|$PY/alzheimers_rf_baseline.py --out-dir $dir --max-depth 10 $sel"
		"rf_venn|$PY/alzheimers_rf_vs_recursieve_venn.py --out-dir $dir"
		"seed_plot|$PY/alzheimers_robustness_plots.py --out-dir $dir"
		"summary|$PY/alzheimers_summary.py --out-dir $dir"
	)
	for step in "${steps[@]}"; do
		local label=${step%%|*} cmd=${step#*|}
		status "$name $label start"
		if $cmd > $ROOT/logs/${name}_${label}.log 2>&1; then
			status "$name $label done"
		else
			status "$name $label FAILED"
			return 1
		fi
	done
	status "$name done"
}

if [ $# -gt 0 ]; then
	run_group "$@"
	exit
fi
run_group excitatory --cell-type Excitatory --n-cells 25000 --balance-groups
run_group inhibitory --cell-type Inhibitory --n-cells 25000 --balance-groups
run_group astrocytes --cell-type Astrocytes --n-cells 7818 --balance-groups
run_group all_cell_types --n-cells 50000
