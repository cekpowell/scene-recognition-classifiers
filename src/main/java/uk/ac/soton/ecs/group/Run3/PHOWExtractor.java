package uk.ac.soton.ecs.group.Run3;

import org.openimaj.feature.DoubleFV;
import org.openimaj.feature.FeatureExtractor;
import org.openimaj.feature.SparseIntFV;
import org.openimaj.image.FImage;
import org.openimaj.image.feature.dense.gradient.dsift.PyramidDenseSIFT;
import org.openimaj.image.feature.local.aggregate.BagOfVisualWords;
import org.openimaj.image.feature.local.aggregate.PyramidSpatialAggregator;
import org.openimaj.ml.clustering.assignment.HardAssigner;
import org.openimaj.util.pair.IntFloatPair;

/**
 * 
 * @author
 */
public class PHOWExtractor implements FeatureExtractor<DoubleFV, FImage> {

    // member variables
    PyramidDenseSIFT<FImage> pdsift;
    HardAssigner<byte[], float[], IntFloatPair> assigner;

    //////////////////
    // INITIALIZING //
    //////////////////

    /**
     * Class constructor.
     * 
     * @param pdsift
     * @param assigner
     */
    public PHOWExtractor(
            PyramidDenseSIFT<FImage> pdsift, HardAssigner<byte[], float[], IntFloatPair> assigner) {
        this.pdsift = pdsift;
        this.assigner = assigner;
    }

    ////////////////////////
    // FEATURE EXTRACTION //
    ////////////////////////

    /**
     * 
     * 
     * @param image
     * @return 
     */
    public DoubleFV extractFeature(FImage image) {
        pdsift.analyseImage(image);

        BagOfVisualWords<byte[]> bovw = new BagOfVisualWords<>(assigner);
        PyramidSpatialAggregator<byte[], SparseIntFV> spatial = new PyramidSpatialAggregator<>(bovw, 2, 4);

        return spatial.aggregate(pdsift.getByteKeypoints(0.015f), image.getBounds()).normaliseFV();
    }
}