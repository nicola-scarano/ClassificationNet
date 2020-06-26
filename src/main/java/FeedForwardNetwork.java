import org.apache.commons.io.FileUtils;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.records.writer.RecordWriter;
import org.datavec.api.records.writer.impl.csv.CSVRecordWriter;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.partition.NumberOfRecordsPartitioner;
import org.datavec.api.split.partition.Partitioner;
import org.datavec.api.transform.TransformProcess;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.writable.Writable;
import org.datavec.local.transforms.LocalTransformExecutor;
import org.deeplearning4j.core.storage.StatsStorage;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.earlystopping.EarlyStoppingConfiguration;
import org.deeplearning4j.earlystopping.EarlyStoppingResult;
import org.deeplearning4j.earlystopping.saver.LocalFileModelSaver;
import org.deeplearning4j.earlystopping.scorecalc.DataSetLossCalculator;
import org.deeplearning4j.earlystopping.termination.MaxEpochsTerminationCondition;
import org.deeplearning4j.earlystopping.termination.MaxTimeIterationTerminationCondition;
import org.deeplearning4j.earlystopping.trainer.EarlyStoppingTrainer;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.model.stats.StatsListener;
import org.deeplearning4j.ui.model.storage.FileStatsStorage;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.ScalarOp;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.KFoldIterator;
import org.nd4j.linalg.dataset.api.iterator.SamplingDataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.TestDataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerMinMaxScaler;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.swing.*;
import java.io.File;
import java.nio.charset.Charset;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.TimeUnit;

//http://localhost:9000/train   for visualization

public class FeedForwardNetwork {

    private static Logger log = LoggerFactory.getLogger(FeedForwardNetwork.class);

    public static void main(String[] args) throws Exception {

        int seed = 123;
        double learningRate = 0.1;
        int batchSize = 50;
        int k=7;

        int numInputs = 9;
        int numOutputs = 2;
        int numHiddenNodes = 40;




        //=========================================================================================
        //                 Step 1: Data preparation
        //=========================================================================================



        //--------------  definizione dello schema  ---------------------

        Schema schema = new Schema.Builder()
                .addColumnsInteger("myIntegerCol_%d",0,9)
                .addColumnCategorical("target", "2", "4")
                .build();

        System.out.println("\n\nOther information obtainable from schema:");
        System.out.println("Number of columns: " + schema.numColumns());
        System.out.println("Column names: " + schema.getColumnNames());
        System.out.println("Column types: " + schema.getColumnTypes());




        //--------------  definizione trasformazioni da eseguire sul dataset  ---------------------

        TransformProcess tp = new TransformProcess.Builder(schema)
                .categoricalToInteger("target")
                .removeColumns("myIntegerCol_0")
                .build();

        //After executing all of these operations, we have a new and different schema:
        Schema outputSchema = tp.getFinalSchema();

        System.out.println("\n\n\nSchema after transforming data:");
        System.out.println(outputSchema);




        //---------------  caricamento dati e esecuzione delle trasformazioni  -----------------

        //Define input and output paths:
        File inputTraining = new File("/Classification/ClassificationNet/Training.csv");
        File outputTraining = new File("/Classification/ClassificationNet/TrainingClean.csv");
        File inputTest = new File("/Classification/ClassificationNet/Test.csv");
        File outputTest = new File("/Classification/ClassificationNet/TestClean.csv");

        if(outputTraining.exists()){
            outputTraining.delete();
        }
        outputTraining.createNewFile();

        if(outputTest.exists()){
            outputTest.delete();
        }
        outputTest.createNewFile();

        //Define input reader and output writer:
        //TRAINING SET
        RecordReader rrTg = new CSVRecordReader(1, ';');
        rrTg.initialize(new FileSplit(inputTraining));

        RecordWriter rwTg = new CSVRecordWriter();
        Partitioner p = new NumberOfRecordsPartitioner();
        rwTg.initialize(new FileSplit(outputTraining), p);

        //TEST SET
        RecordReader rrTs = new CSVRecordReader(1, ';');
        rrTs.initialize(new FileSplit(inputTest));

        RecordWriter rwTs = new CSVRecordWriter();
        Partitioner p1 = new NumberOfRecordsPartitioner();
        rwTs.initialize(new FileSplit(outputTest), p1);


        //Process the data:
        List<List<Writable>> originalTrainingData = new ArrayList<>();
        while(rrTg.hasNext()){
            originalTrainingData.add(rrTg.next());
        }

        List<List<Writable>> originalTestData = new ArrayList<>();
        while(rrTs.hasNext()){
            originalTestData.add(rrTs.next());
        }

        System.out.println("\n\n---- Original Training Data File ----");
        String fileContents = FileUtils.readFileToString(inputTraining, Charset.defaultCharset());
        System.out.println(fileContents);

        System.out.println("\n\n---- Original Test Data File ----");
        fileContents = FileUtils.readFileToString(inputTest, Charset.defaultCharset());
        System.out.println(fileContents);

        List<List<Writable>> processedTrainingData = LocalTransformExecutor.execute(originalTrainingData, tp);
        rwTg.writeBatch(processedTrainingData);


        List<List<Writable>> processedTestData = LocalTransformExecutor.execute(originalTestData, tp);
        rwTs.writeBatch(processedTestData);

        rwTs.close();
        rwTg.close();



        //---------------      NORMALIZZAZIONE DEI DATI     -----------------------

        RecordReader rrTraining = new CSVRecordReader(0, ',');
        rrTraining.initialize(new FileSplit(outputTraining));
        DataSetIterator trainingIter = new RecordReaderDataSetIterator(rrTraining,batchSize,9,2);
        DataSet training = trainingIter.next();
        KFoldIterator KFtrainingIter = new KFoldIterator(k,training);   //CROSS validation iterator

        RecordReader rrTest = new CSVRecordReader(0, ',');
        rrTest.initialize(new FileSplit(outputTest));
        DataSetIterator testIter = new RecordReaderDataSetIterator(rrTest,1,9,2);
        DataSet test = testIter.next();

        NormalizerMinMaxScaler preProcessor = new NormalizerMinMaxScaler(0,1);
        preProcessor.fit(training);

        log.info("First ten Training set before normalization");
        log.info("\n{}",training.getRange(0,9));
        preProcessor.transform(training);
        log.info("First ten Training set after normalization");
        log.info("\n{}",training.getRange(0,9));

        log.info("First Test set before normalization");
        log.info("\n{}",test.getRange(0,1));
        preProcessor.transform(test);
        log.info("First Test set after normalization");
        log.info("\n{}",test.getRange(0,1));







        //=================================================================================================
        //                 Step 2: Neural network configuration, training and evaluation
        //=================================================================================================

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .weightInit(WeightInit.XAVIER)
                .updater(new Nesterovs(learningRate, 0.9))
                .list()
                .layer(new DenseLayer.Builder().nIn(numInputs).nOut(numHiddenNodes)
                        .activation(Activation.RELU)
                        .build())
                .layer(new DenseLayer.Builder().nIn(numHiddenNodes).nOut(numHiddenNodes)
                        .activation(Activation.RELU)
                        .build())
                .layer(new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)//XENT loss function used for binary classification
                        .activation(Activation.SOFTMAX)
                        .nIn(numHiddenNodes).nOut(numOutputs).build())
                .build();





        batchSize = 5000; // Grande in maniera tale che ogni epoca di addestramento corrisponda ad un batch
        ArrayList<MultiLayerNetwork> models = new ArrayList<>();
        while (KFtrainingIter.hasNext()) {
            DataSet trdataset = KFtrainingIter.next();
            DataSet tedataset = KFtrainingIter.testFold();

            EarlyStoppingConfiguration esConf = new EarlyStoppingConfiguration.Builder()
                    .epochTerminationConditions(new MaxEpochsTerminationCondition(500))
                    .iterationTerminationConditions(new MaxTimeIterationTerminationCondition(20, TimeUnit.MINUTES))
                    .scoreCalculator(new DataSetLossCalculator(new TestDataSetIterator(tedataset), true))
                    .evaluateEveryNEpochs(1)
                    .modelSaver(new LocalFileModelSaver("/bestmodel"))
                    .build();
            EarlyStoppingTrainer trainer = new EarlyStoppingTrainer(esConf,conf, new SamplingDataSetIterator(trdataset, batchSize, trdataset.getLabels().rows()));

            //Conduct early stopping training:
            EarlyStoppingResult <MultiLayerNetwork> result = trainer.fit();

            MultiLayerNetwork model = result.getBestModel();

            //Print out the results:
            System.out.println("Termination reason: " + result.getTerminationReason());
            System.out.println("Termination details: " + result.getTerminationDetails());
            System.out.println("Total epochs: " + result.getTotalEpochs());
            System.out.println("Best epoch number: " + result.getBestModelEpoch());
            System.out.println("Score at best epoch: " + result.getBestModelScore());
            System.out.println("EPOCA : SCORE");
            Map<Integer,Double> scoreVsIter = result.getScoreVsEpoch();
            scoreVsIter.forEach((key, value) -> System.out.println(key + ":" + value));

            models.add(model.clone());
        }





        //evaluation
        System.out.println("Evaluate model....");
        Evaluation eval = new Evaluation(numOutputs);
        while(testIter.hasNext()){
            DataSet t = testIter.next();
            INDArray features = t.getFeatures();
            INDArray lables = t.getLabels();
            System.out.println("Valore esatto"+ lables);
            INDArray predicted = Nd4j.zeros(1, 2);
            int i=0;
            while(i<models.size()){
                System.out.println("Valore calcolato da modello: " + i + " = "+ models.get(i).output(features,false));
                predicted = predicted.add(models.get(i).output(features,false).castTo(DataType.FLOAT));
                System.out.println("sum " + predicted);
                i++;
            }
            predicted= predicted.div(k);

            System.out.println("Predizione: "+ predicted);
            eval.eval(lables, predicted);
        }

        System.out.println(eval.stats());

    }




}
