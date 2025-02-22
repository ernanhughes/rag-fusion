# Get the list of Ollama models
$ollamaList = ollama list

# Split the output into lines and skip the header
$models = $ollamaList | Select-Object -Skip 1

# Loop through each model and pull it
foreach ($model in $models) {
    # Split the line by spaces and get the model name (first column)
    $modelName = ($model -split '\s+')[0]

    # Check if the model name contains "reviewer" (case-insensitive)
    if ($modelName -notmatch "reviewer") {
        # Pull the model
        Write-Host "Pulling model: $modelName"
        ollama pull $modelName
    }
}