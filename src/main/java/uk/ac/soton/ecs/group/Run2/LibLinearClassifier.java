package uk.ac.soton.ecs.group.Run2;

import de.bwaldvogel.liblinear.SolverType;
import org.openimaj.data.dataset.*;
import org.openimaj.experiment.dataset.sampling.GroupedUniformRandomisedSampler;
import org.openimaj.image.FImage;
import org.openimaj.ml.annotation.Annotator;
import org.openimaj.ml.annotation.linear.LiblinearAnnotator;
import org.openimaj.ml.clustering.DoubleCentroidsResult;
import org.openimaj.ml.clustering.assignment.HardAssigner;
import org.openimaj.ml.clustering.kmeans.DoubleKMeans;
import org.openimaj.util.pair.IntDoublePair;
import uk.ac.soton.ecs.group.MyClassifier;

import java.util.ArrayList;
import java.util.List;

/**
 * 
 * 
 * @author
 */
public class LibLinearClassifier extends MyClassifier {

    // member variables
    private final DSPPExtractor extractor;
    private final LiblinearAnnotator<FImage, String> annotator;
    private final int K;
    private final int patchSize;
    private final int patchEvery;

    //////////////////
    // INITIALIZING //
    //////////////////

    /**
     * Class constructor.
     * 
     * @param trainingData
     * @param patchSize
     * @param patchEvery
     * @param K
     * @param sampleSize
     */
    public LibLinearClassifier(GroupedDataset trainingData, int patchSize, int patchEvery, int K, int sampleSize) {
        this.K = K;
        this.patchSize = patchSize;
        this.patchEvery = patchEvery;

        HardAssigner<double[], double[], IntDoublePair> assigner = trainQuantiser(GroupedUniformRandomisedSampler.sample(trainingData, sampleSize));
        this.extractor = new DSPPExtractor(assigner, patchSize, patchEvery);
        this.annotator = new LiblinearAnnotator<>(this.extractor, LiblinearAnnotator.Mode.MULTICLASS, SolverType.L2R_L2LOSS_SVC, 1.0, 0.00001);

        this.fit(trainingData);
    }

    //////////////
    // TRAINING //
    //////////////

    /**
     * Fits the Linear Classifier to the given training data.
     * 
     * @param trainingData The training data the Linear Classifier will
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
     * @return
     */
    private HardAssigner<double[], double[], IntDoublePair> trainQuantiser(Dataset<FImage> sample) {
        List<double[]> allkeys = new ArrayList<>();
        for (FImage img : sample) allkeys.addAll(DSPPExtractor.getPatches(img, patchSize, patchEvery));

        DoubleKMeans km = DoubleKMeans.createKDTreeEnsemble(K);
        DoubleCentroidsResult result = km.cluster(allkeys.toArray(new double[0][]));

        return result.defaultHardAssigner();
    }

    //////////////////////////
    // GETTERS AND SETTERS  //
    //////////////////////////

    @Override
    public Annotator getClassifier() {
        return this.annotator;
    }
}