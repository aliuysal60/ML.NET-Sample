using Microsoft.ML.Runtime.Api;

namespace ML.NET_Sample
{
    public class SalaryData
    {
        [Column("0")]
        public float YILLAR;

        [Column("1", name: "Label")]
        public float MAAS;
    }
}
