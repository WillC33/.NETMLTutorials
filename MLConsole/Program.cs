// See https://aka.ms/new-console-template for more information
using Microsoft.ML;
using MLConsole;

// ====================================================================
// REGRESSION MODEL

//TaxiFareModel taxiFareModel = new();

// ====================================================================
// MULTI CATEGORY CLASSIFIER

//SupportIssueModel supportIssueModel = new();
//GitHubIssue singleIssue = new() { Title = "Entity Framework crashes", Description = "When connecting to the database, EF is crashing" };
//GitHubIssue singleIssue2 = new() { Title = "Logging isn't working", Description = "The logs in my output are all blank" };
//GitHubIssue singleIssue3 = new() { Title = "My cookies aren't storing in my browser so the responses are all 401", Description = "When I log in I am redirected to the login screen over and over again. I've looked at the cookies that are being stored and they do not ever appear in my localstorage" };

//supportIssueModel.PredictIssue(singleIssue);
//supportIssueModel.PredictIssue(singleIssue2);
//supportIssueModel.PredictIssue(singleIssue3);

for(int i = 0; i <= 50; i++)
{
    KMeansClustering iris = new();
}

Console.WriteLine("Exit");