using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Models;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms;
using ML.NET_Sample.Data_Models;
using System;

namespace ML.NET_Sample
{
    class Program
    {
        private const int TAHMIN_EDILECEK_YIL = 10;

        static void Main(string[] args)
        {
            var dataset = @"C:\MLNET\MAASVeri.csv";
            var testDataset = @"C:\MLNET\MAASVeri-test.csv";

            var pipeline = new LearningPipeline
            {
                new TextLoader(dataset).CreateFrom<SalaryData>(useHeader: true, separator: ','),
                new ColumnConcatenator("Features", "YILLAR"),
                new GeneralizedAdditiveModelRegressor()
            };

            Console.WriteLine("..........Eğitiliyor..........");
            var model = pipeline.Train<SalaryData, SalaryPrediction>();

            model.WriteAsync(@"C:\MLNET\model.zip");

            Console.WriteLine(Environment.NewLine);
            Console.WriteLine("..........Değerlendiriliyor..........");
            var testData = new TextLoader(testDataset).CreateFrom<SalaryData>(useHeader: true, separator: ',');

            var evaluator = new RegressionEvaluator();
            var metrics = evaluator.Evaluate(model, testData);

            Console.WriteLine($"Karekök Ortalama: {metrics.Rms}");
            Console.WriteLine($"R^2: {metrics.RSquared}");

            Console.WriteLine(Environment.NewLine);
            Console.WriteLine("..........Tahmin..........");
            var prediction = model.Predict(new SalaryData { YILLAR = TAHMIN_EDILECEK_YIL });

            Console.WriteLine($"{TAHMIN_EDILECEK_YIL} yıl sonraki maaşınız {String.Format("{0:C}", prediction.PredictedSalary)}");

            Console.ReadLine();
        }
    }
}
