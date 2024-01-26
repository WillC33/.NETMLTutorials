using Microsoft.ML;

namespace MLConsole;

public class KMeansClustering
{
    string _dataPath = Path.Combine(Environment.CurrentDirectory, "TrainingData", "iris.data");
    string _modelPath = Path.Combine(Environment.CurrentDirectory, "Outputs", "IrisClusteringModel.zip");
    private MLContext _mlContext;
    
    public KMeansClustering()
    {
        _mlContext = new(seed: 0);
        IDataView dataView = _mlContext.Data.LoadFromTextFile<IrisDataModel>(_dataPath, hasHeader: false, separatorChar: ',');
        
        string featuresColumnName = "Features";
        var pipeline = _mlContext.Transforms
            .Concatenate(featuresColumnName, "SepalLength", "SepalWidth", "PetalLength", "PetalWidth")
            .Append(_mlContext.Clustering.Trainers.KMeans(featuresColumnName, numberOfClusters: 3));
        
        var model = pipeline.Fit(dataView);

        using FileStream fileStream = new(_modelPath, FileMode.Create, FileAccess.Write, FileShare.Write);
        _mlContext.Model.Save(model, dataView.Schema, fileStream);
        
        //Predicts
        IrisDataModel data = new() //Setosa
        {
            SepalLength = 5.1f,
            SepalWidth = 3.5f,
            PetalLength = 1.4f,
            PetalWidth = 0.2f
        };
        
        var predictor = _mlContext.Model.CreatePredictionEngine<IrisDataModel, ClusterPrediction>(model);
        var prediction = predictor.Predict(data);
        Console.WriteLine($"Cluster: {prediction.PredictedClusterId}"); // This isn't very accurate and multiple training instances create very different and inconsistent results
        Console.WriteLine($"Distances: {string.Join(" ", prediction.Distances ?? Array.Empty<float>())}");
    }
}