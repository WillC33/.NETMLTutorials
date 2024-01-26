using Microsoft.ML;

namespace MLConsole;

public class SupportIssueModel
{
    string _trainDataPath = Path.Combine(Environment.CurrentDirectory, "TrainingData", "issues_train.tsv.txt");
    string _testDataPath = Path.Combine(Environment.CurrentDirectory, "TrainingData", "issues_test.tsv.txt");
    string _modelPath = Path.Combine(Environment.CurrentDirectory, "Outputs", "Model.zip");
    MLContext _mlContext;
    IDataView _trainingDataView;
    PredictionEngine<GitHubIssue, IssuePrediction> _predEngine;
    ITransformer _trainedModel;

    public SupportIssueModel()
    {
        _mlContext = new MLContext();
        _trainingDataView = _mlContext.Data.LoadFromTextFile<GitHubIssue>(_trainDataPath, hasHeader: true);
        
        var pipeline = ProcessData();
        var trainingPipeline = BuildAndTrainModel(_trainingDataView, pipeline);
        Evaluate(_trainingDataView.Schema);
        SaveModelAsFile(_mlContext, _trainingDataView.Schema, _trainedModel);
    }

    private IEstimator<ITransformer> ProcessData()
    {
        return _mlContext.Transforms.Conversion.MapValueToKey(inputColumnName: "Area", outputColumnName: "Label")
            .Append(_mlContext.Transforms.Text.FeaturizeText(inputColumnName: "Title",
                outputColumnName: "TitleFeaturized"))
            .Append(_mlContext.Transforms.Text.FeaturizeText(inputColumnName: "Description",
                outputColumnName: "DescriptionFeaturized"))
            .Append(_mlContext.Transforms.Concatenate("Features", "TitleFeaturized", "DescriptionFeaturized"))
            .AppendCacheCheckpoint(_mlContext);
    }
    
    IEstimator<ITransformer> BuildAndTrainModel(IDataView trainingDataView, IEstimator<ITransformer> pipeline)
    {
        var trainingPipeline = pipeline.Append(_mlContext.MulticlassClassification.Trainers.SdcaMaximumEntropy("Label", "Features"))
            .Append(_mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));
        
        _trainedModel = trainingPipeline.Fit(trainingDataView);
        _predEngine = _mlContext.Model.CreatePredictionEngine<GitHubIssue, IssuePrediction>(_trainedModel);
        
        GitHubIssue issue = new GitHubIssue() {
            Title = "WebSockets communication is slow in my machine",
            Description = "The WebSockets communication used under the covers by SignalR looks like is going slow in my development machine.."
        };
        
        var prediction = _predEngine.Predict(issue);
        Console.WriteLine($"=============== Single Prediction just-trained-model - Result: {prediction.Area} ===============");
        return trainingPipeline;
    }

    void Evaluate(DataViewSchema trainingDataViewSchema)
    {
        var testDataView = _mlContext.Data.LoadFromTextFile<GitHubIssue>(_testDataPath, hasHeader: true);
        var testMetrics = _mlContext.MulticlassClassification.Evaluate(_trainedModel.Transform(testDataView));

        Console.WriteLine(
            $"*************************************************************************************************************");
        Console.WriteLine($"*       Metrics for Multi-class Classification model - Test Data     ");
        Console.WriteLine(
            $"*------------------------------------------------------------------------------------------------------------");
        Console.WriteLine($"*       MicroAccuracy:    {testMetrics.MicroAccuracy:0.###}");
        Console.WriteLine($"*       MacroAccuracy:    {testMetrics.MacroAccuracy:0.###}");
        Console.WriteLine($"*       LogLoss:          {testMetrics.LogLoss:#.###}");
        Console.WriteLine($"*       LogLossReduction: {testMetrics.LogLossReduction:#.###}");
        Console.WriteLine(
            $"*************************************************************************************************************");
    }
    
    void SaveModelAsFile(MLContext mlContext,DataViewSchema trainingDataViewSchema, ITransformer model)
    {
        mlContext.Model.Save(model, trainingDataViewSchema, _modelPath);
    }
    
    public void PredictIssue(GitHubIssue issue)
    {
        ITransformer loadedModel = _mlContext.Model.Load(_modelPath, out var modelInputSchema);
      
        _predEngine = _mlContext.Model.CreatePredictionEngine<GitHubIssue, IssuePrediction>(loadedModel);
        var prediction = _predEngine.Predict(issue);
        Console.WriteLine($"=============== Single Prediction - Result: {prediction.Area} ===============");
    }
}