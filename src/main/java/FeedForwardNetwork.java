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
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerMinMaxScaler;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.nio.charset.Charset;
import java.util.ArrayList;
import java.util.List;

public class FeedForwardNetwork {

    private static Logger log = LoggerFactory.getLogger(FeedForwardNetwork.class);

    public static void main(String[] args) throws Exception {

        int seed = 123;
        double learningRate = 0.01;
        int batchSize = 50;
        int nEpochs = 30;

        int numInputs = 10;
        int numOutputs = 2;
        int numHiddenNodes = 20;



        //--------------  definizione dello schema  ---------------------


        Schema schema = new Schema.Builder()
                .addColumnsInteger("myIntegerCol_%d",0,9)
                .addColumnCategorical("target", "2", "4")
                .build();

        System.out.println("\n\nOther information obtainable from schema:");
        System.out.println("Number of columns: " + schema.numColumns());
        System.out.println("Column names: " + schema.getColumnNames());
        System.out.println("Column types: " + schema.getColumnTypes());




        //--------------  definizione operazioni da fare  ---------------------

        TransformProcess tp = new TransformProcess.Builder(schema)
                .categoricalToInteger("target")
                .removeColumns("myIntegerCol_0")
                .build();

        //After executing all of these operations, we have a new and different schema:
        Schema outputSchema = tp.getFinalSchema();

        System.out.println("\n\n\nSchema after transforming data:");
        System.out.println(outputSchema);



        //---------------  caricamento dati e esecuzione delle operazioni  -----------------

        //Define input and output paths:
        File inputFile = new File("/Classification/ClassificationNet/Training.csv");
        File outputFile = new File("/Classification/ClassificationNet/TrainingClean.csv");
        if(outputFile.exists()){
            outputFile.delete();
        }
        outputFile.createNewFile();


        //Define input reader and output writer:
        RecordReader rr = new CSVRecordReader(1, ';');
        rr.initialize(new FileSplit(inputFile));

        RecordWriter rw = new CSVRecordWriter();
        Partitioner p = new NumberOfRecordsPartitioner();
        rw.initialize(new FileSplit(outputFile), p);


        //Process the data:
        List<List<Writable>> originalData = new ArrayList<>();
        while(rr.hasNext()){
            originalData.add(rr.next());
        }

        System.out.println("\n\n---- Original Data File ----");
        String originalFileContents = FileUtils.readFileToString(inputFile, Charset.defaultCharset());
        System.out.println(originalFileContents);

        List<List<Writable>> processedData = LocalTransformExecutor.execute(originalData, tp);
        rw.writeBatch(processedData);
        rw.close();

        System.out.println("\n\n---- Processed Data File ----");
        String fileContents = FileUtils.readFileToString(outputFile, Charset.defaultCharset());
        System.out.println(fileContents);




        //---------------      NORMALIZZAZIONE DEI DATI     -----------------------
        RecordReader rrTest = new CSVRecordReader(0, ',');
        rrTest.initialize(new FileSplit(outputFile));
        DataSetIterator testIter = new RecordReaderDataSetIterator(rrTest,batchSize,9,2);
        DataSet test = testIter.next();
        NormalizerMinMaxScaler preProcessor = new NormalizerMinMaxScaler(0,1);
        preProcessor.fit(test);
        log.info("First ten before normalization");
        log.info("\n{}",test.getRange(0,9));
        preProcessor.transform(test);
        log.info("First ten after normalization");
        log.info("\n{}",test.getRange(0,9));














        /*
        //Load the training data:
        RecordReader rr = new CSVRecordReader(1,';','"');
        //RecordReader è un'interfaccia implementata (ereditata) dalle classi padre di CSVRecordReader;
          CSVRecordReader() costruttore: divisore: ','
                                         skip:'0'
                                         testo:'"'

        rr.initialize(new FileSplit(new File("ClassificationNet/Training.csv"))); inizialize è utilizzato per l'inizializzazione del file
        DataSetIterator trainIter = new RecordReaderDataSetIterator(rr,batchSize,10,2);
        //batchSize: numero di esempi da utilizzare
        DataSet train = trainIter.next();

        //Load the test/evaluation data:
        RecordReader rrTest = new CSVRecordReader(1,';','"');
        rrTest.initialize(new FileSplit(new File("ClassificationNet/Test.csv")));
        DataSetIterator testIter = new RecordReaderDataSetIterator(rrTest,batchSize,10,2);
        DataSet test = testIter.next();*/

/*
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .weightInit(WeightInit.XAVIER)
                .updater(new Nesterovs(learningRate, 0.9))
                .list()
                .layer(new DenseLayer.Builder().nIn(numInputs).nOut(numHiddenNodes)
                        .activation(Activation.RELU)
                        .build())
                .layer(new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)//XENT loss function used for binary classification
                        .activation(Activation.SOFTMAX)
                        .nIn(numHiddenNodes).nOut(numOutputs).build())
                .build();

        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        model.setListeners(new ScoreIterationListener(10));

        model.fit( trainIter, nEpochs );

        System.out.println("Evaluate model....");
        Evaluation eval = new Evaluation(numOutputs);
        while(testIter.hasNext()){
            DataSet t = testIter.next();
            INDArray features = t.getFeatures();
            INDArray lables = t.getLabels();
            INDArray predicted = model.output(features,false);
            eval.eval(lables, predicted);

        }

        //Print the evaluation statistics
        System.out.println(eval.stats());*/

    }

}
