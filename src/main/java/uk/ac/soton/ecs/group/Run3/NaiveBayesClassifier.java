package uk.ac.soton.ecs.group.Run3;

import org.openimaj.data.DataSource;
import org.openimaj.data.dataset.Dataset;
import org.openimaj.data.dataset.GroupedDataset;
import org.openimaj.experiment.dataset.sampling.GroupedUniformRandomisedSampler;
import org.openimaj.feature.local.data.LocalFeatureListDataSource;
import org.openimaj.feature.local.list.LocalFeatureList;
import org.openimaj.image.FImage;
import org.openimaj.image.feature.dense.gradient.dsift.ByteDSIFTKeypoint;
import org.openimaj.image.feature.dense.gradient.dsift.DenseSIFT;
import org.openimaj.image.feature.dense.gradient.dsift.PyramidDenseSIFT;
import org.openimaj.ml.annotation.Annotator;
import org.openimaj.ml.annotation.bayes.NaiveBayesAnnotator;
import org.openimaj.ml.clustering.ByteCentroidsResult;
import org.openimaj.ml.clustering.assignment.HardAssigner;
import org.openimaj.ml.clustering.kmeans.ByteKMeans;
import org.openimaj.util.pair.IntFloatPair;
import uk.ac.soton.ecs.group.MyClassifier;

import java.util.ArrayList;
import java.util.List;

/**
 * 
 * 
 * @author
 */
public class NaiveBayesClassifier extends MyClassifier {

    // member variables
    private final PHOWExtractor extractor;
    private final NaiveBayesAnnotator annotator;
    private final int K;

    //////////////////
    // INITIALIZING //
    //////////////////

    /**
     * Class constructor.
     * 
     * @param trainingData
     * @param K
     * @param sampleSize
     */
    public NaiveBayesClassifier(GroupedDataset trainingData, int K, int sampleSize) {
        this.K = K;

        DenseSIFT dsift = new DenseSIFT(3, 7);
        PyramidDenseSIFT<FImage> pdsift = new PyramidDenseSIFT<>(dsift, 6f, 4,6,8,10);
        HardAssigner<byte[], float[], IntFloatPair> assigner = trainQuantiser(GroupedUniformRandomisedSampler.sample(trainingData, sampleSize), pdsift);

        this.extractor = new PHOWExtractor(pdsift, assigner);
        this.annotator = new NaiveBayesAnnotator<>(extractor, NaiveBayesAnnotator.Mode.MAXIMUM_LIKELIHOOD);
        this.fit(trainingData);
    }

    //////////////
    // TRAINING //
    //////////////

    /**
     * Fits the Naive Bayes classifier to the given training data.
     * 
     * @param trainingData The training data the Naive Bayes classifier will
     * be fit to.
     */
    @Override
    public void fit(GroupedDataset trainingData) {
        this.annotator.train(trainingData);
    }

    /**
     * 
     * 
     * @param sample
     * @param pdsift
     * @return
     */
    private HardAssigner<byte[], float[], IntFloatPair> trainQuantiser(
        Dataset<FImage> sample, PyramidDenseSIFT<FImage> pdsift) {
        List<LocalFeatureList<ByteDSIFTKeypoint>> allkeys = new ArrayList<>();

        for (FImage img : sample) {
            pdsift.analyseImage(img);
            allkeys.add(pdsift.getByteKeypoints(0.005f));
        }

        ByteKMeans km = ByteKMeans.createKDTreeEnsemble(this.K);
        DataSource<byte[]> datasource = new LocalFeatureListDataSource<>(allkeys);
        ByteCentroidsResult result = km.cluster(datasource);

        return result.defaultHardAssigner();
    }   

    /////////////////////////
    // GETTERS AND SETTERS //
    /////////////////////////

    @Override
    public Annotator getClassifier() {
        return this.annotator;
    }
}