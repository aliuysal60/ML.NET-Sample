using Microsoft.ML.Runtime.Api;

namespace ML.NET_Sample.Data_Models
{
    public class SalaryPrediction
    {
        [ColumnName("Score")]
        public float PredictedSalary;
    }
}
