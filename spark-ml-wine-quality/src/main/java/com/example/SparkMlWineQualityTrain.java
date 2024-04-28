package com.example;


import java.io.IOException;
import java.util.Arrays;

import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.feature.StandardScaler;
import org.apache.spark.ml.classification.LogisticRegression;
import org.apache.spark.ml.classification.RandomForestClassifier;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.tuning.ParamGridBuilder;
import org.apache.spark.ml.tuning.CrossValidator;
import org.apache.spark.ml.tuning.CrossValidatorModel;
import org.apache.spark.ml.param.ParamMap;
import org.apache.spark.ml.classification.LogisticRegressionModel;
import org.apache.spark.ml.classification.RandomForestClassificationModel;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.ml.Pipeline;



public class SparkMlWineQualityTrain {
    public static void main(String[] args) throws IOException {
        SparkSession spark = SparkSession.builder().appName("Spark ML Wine Quality Train").getOrCreate();

        // Load training data
        Dataset<Row> df = spark.read().option("header", "true").option("sep", ";").option("inferSchema", "true")
                .csv("s3://spark-ml-wine-quality/TrainingDataset.csv");

        String[] columns = new String[] { "fixed_acidity", "volatile_acidity", "citric_acid", "residual_sugar",
                "chlorides", "free_sulfur_dioxide", "total_sulfur_dioxide", "density", "pH", "sulphates", "alcohol",
                "quality" };
        df = df.toDF(columns);

        VectorAssembler assembler = new VectorAssembler().setInputCols(Arrays.copyOfRange(columns, 0, columns.length - 1))
                .setOutputCol("features");
        StandardScaler scaler = new StandardScaler().setInputCol("features").setOutputCol("scaledFeatures")
                .setWithStd(true).setWithMean(true);

        Pipeline pipeline = new Pipeline().setStages(new PipelineStage[] { assembler, scaler });
        PipelineModel pipelineModel = pipeline.fit(df);
        Dataset<Row> trainData = pipelineModel.transform(df);

        //Load validation data
        df = spark.read().option("header", "true").option("sep", ";").option("inferSchema", "true")
                .csv("s3://spark-ml-wine-quality/ValidationDataset.csv");
        df = df.toDF(columns);

        PipelineModel pipelineModelValidation = pipeline.fit(df);
        Dataset<Row> validationData = pipelineModelValidation.transform(df);


        // Logistic Regression
        LogisticRegression lr = new LogisticRegression().setLabelCol("quality").setFeaturesCol("scaledFeatures")
                .setMaxIter(10).setRegParam(0.3);
        LogisticRegressionModel lrModel = lr.fit(trainData);

        // Evaluate Logistic Regression
        MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator().setLabelCol("quality")
                .setPredictionCol("prediction");
        double trainAccuracy = evaluator.evaluate(lrModel.transform(trainData));
        double validationAccuracy = evaluator.evaluate(lrModel.transform(validationData)
        );

        System.out.println("Train Accuracy for linear regression: " + trainAccuracy + "\nValidation Accuracy for linear regression: " + validationAccuracy);

        // Random Forest with CV
        RandomForestClassifier rf = new RandomForestClassifier().setLabelCol("quality").setFeaturesCol("scaledFeatures");

        ParamGridBuilder paramGridBuilder = new ParamGridBuilder();
        ParamMap[] paramGrid = paramGridBuilder.addGrid(rf.numTrees(), new int[] { 20, 50, 100 })
                .addGrid(rf.maxDepth(), new int[] { 5, 10, 15 }).build();

        CrossValidator cv = new CrossValidator().setEstimator(rf).setEvaluator(evaluator).setEstimatorParamMaps(paramGrid)
                .setNumFolds(3);

        CrossValidatorModel cvModel = cv.fit(trainData);
        double bestScore = cvModel.avgMetrics()[0];

        System.out.println("Best Validation F1 Score: " + bestScore);

        String bestModelPath = "s3://spark-ml-wine-quality/best_model";
        RandomForestClassificationModel bestRfModel = (RandomForestClassificationModel) cvModel.bestModel();
        bestRfModel.write().overwrite().save(bestModelPath);
        System.out.println("Best model saved to " + bestModelPath);

        spark.stop();
    }
}
