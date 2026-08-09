param(
    [string]$SignalRecord = "tpehgt_p001",
    [ValidateSet("ehg1", "ehg2", "ehg3")]
    [string]$SignalChannel = "ehg2",
    [int]$SignalSegmentId = 0
)

$ErrorActionPreference = "Stop"

function Invoke-PythonStep {
    param(
        [Parameter(Mandatory = $true)]
        [string[]]$Arguments
    )

    & python @Arguments

    if ($LASTEXITCODE -ne 0) {
        throw "Python command failed: python $($Arguments -join ' ')"
    }
}

Invoke-PythonStep -Arguments @("io_readers.py")
Invoke-PythonStep -Arguments @("extract_features.py")
Invoke-PythonStep -Arguments @("classify_groupwise_cv.py", "--no-resume")
Invoke-PythonStep -Arguments @(
    "grouped_permutation_importance.py",
    "--n-repeats", "30",
    "--permutations-per-fold", "1"
)
Invoke-PythonStep -Arguments @(
    "plot_signal_figures.py",
    "--record", $SignalRecord,
    "--channel", $SignalChannel,
    "--segment-id", $SignalSegmentId.ToString()
)
Invoke-PythonStep -Arguments @("plot_performance_figures.py")
Invoke-PythonStep -Arguments @("plot_feature_distributions.py")
