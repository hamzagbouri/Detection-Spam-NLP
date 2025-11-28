# custom_transformers.py
import nltk
from pyspark.ml import Transformer
from pyspark.ml.param.shared import Param, Params, TypeConverters
from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable
from pyspark.sql import functions as F, types as T

class StemmerTransformer(Transformer, DefaultParamsReadable, DefaultParamsWritable):
    inputCol = Param(Params._dummy(), "inputCol", "input column", TypeConverters.toString)
    outputCol = Param(Params._dummy(), "outputCol", "output column", TypeConverters.toString)

    def __init__(self, inputCol=None, outputCol=None):
        super().__init__()
        self._setDefault(inputCol="filtered_tokens", outputCol="stemmed")
        if inputCol:
            self._set(inputCol=inputCol)
        if outputCol:
            self._set(outputCol=outputCol)

    def _transform(self, dataset):
        stemmer = nltk.stem.PorterStemmer()

        def stem_tokens(tokens):
            if tokens is None:
                return []
            return [stemmer.stem(str(t)) for t in tokens if t]

        stem_udf = F.udf(stem_tokens, T.ArrayType(T.StringType()))

        return dataset.withColumn(
            self.getOrDefault(self.outputCol),
            stem_udf(F.col(self.getOrDefault(self.inputCol)))
        )