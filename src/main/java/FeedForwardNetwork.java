import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
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
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;

public class FeedForwardNetwork {

    private static Logger log = LoggerFactory.getLogger(FeedForwardNetwork.class);

    public static void main(String[] args) throws Exception {

        int seed = 123;
        double learningRate = 0.01;
        int batchSize = 50;
        int nEpochs = 30;

        int numInputs = 10;
        int numOutputs = 1;
        int numHiddenNodes = 20;


        //Load the training data:
        RecordReader rr = new CSVRecordReader(1,';','"');
        /*RecordReader è un'interfaccia implementata (ereditata) dalle classi padre di CSVRecordReader;
          CSVRecordReader() costruttore: divisore: ','
                                         skip:'0'
                                         testo:'"'
         */
        rr.initialize(new FileSplit(new File("Training.csv"))); /*inizialize è utilizzato per l'inizializzazione del file */
        DataSetIterator trainIter = new RecordReaderDataSetIterator(rr,batchSize,10,1);
        /* batchSize: numero di esempi da utilizzare*/
        DataSet train = trainIter.next();

        //Load the test/evaluation data:
        RecordReader rrTest = new CSVRecordReader(1,';','"');
        rrTest.initialize(new FileSplit(new File("Test.csv")));
        DataSetIterator testIter = new RecordReaderDataSetIterator(rrTest,batchSize,10,1);
        DataSet test = testIter.next();


        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .weightInit(WeightInit.XAVIER)
                .updater(new Nesterovs(learningRate, 0.9))
                .list()
                .layer(new DenseLayer.Builder().nIn(numInputs).nOut(numHiddenNodes)
                        .activation(Activation.SIGMOID)
                        .build())
                .layer(new OutputLayer.Builder(LossFunctions.LossFunction.XENT)//XENT loss function used for binary classification
                        .activation(Activation.SIGMOID)
                        .nIn(numHiddenNodes).nOut(numOutputs).build())
                .build();

        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        model.setListeners(new ScoreIterationListener(10));

        model.fit( trainIter, nEpochs );

        //evaluate the model on the test set
        Evaluation eval = new Evaluation(2);
        INDArray output = model.output(test.getFeatures());
        eval.eval(test.getLabels(), output);
        log.info(eval.stats());

        //Print the evaluation statistics
        System.out.println(eval.stats());

    }

}
