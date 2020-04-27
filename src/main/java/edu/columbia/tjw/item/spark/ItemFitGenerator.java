package edu.columbia.tjw.item.spark;

import edu.columbia.tjw.item.*;
import edu.columbia.tjw.item.algo.QuantileBreakdown;
import edu.columbia.tjw.item.base.SimpleRegressor;
import edu.columbia.tjw.item.base.SimpleStatus;
import edu.columbia.tjw.item.base.StandardCurveFactory;
import edu.columbia.tjw.item.base.StandardCurveType;
import edu.columbia.tjw.item.base.raw.RawFittingGrid;
import edu.columbia.tjw.item.fit.EntropyCalculator;
import edu.columbia.tjw.item.fit.FitResult;
import edu.columbia.tjw.item.fit.GradientResult;
import edu.columbia.tjw.item.fit.ItemFitter;
import edu.columbia.tjw.item.fit.calculator.FitPoint;
import edu.columbia.tjw.item.fit.calculator.FitPointAnalyzer;
import edu.columbia.tjw.item.optimize.ConvergenceException;
import edu.columbia.tjw.item.optimize.OptimizationTarget;
import edu.columbia.tjw.item.util.MathTools;

import java.io.*;
import java.util.*;

public final class ItemFitGenerator
{
    private final File _fitFolder;
    private final RawFittingGrid<SimpleStatus, SimpleRegressor> _dataFit;
    private final RawFittingGrid<SimpleStatus, SimpleRegressor> _dataTest;
    private final File _settingsFile;
    private final long _startingSeed = 0x0afebabe;


    private ItemFitGenerator(final File fitFolder_) throws IOException
    {
        _fitFolder = fitFolder_;
        _settingsFile = new File(_fitFolder, "classifier_settings.dat");
        final File fitData = new File(fitFolder_, "data_fit.dat");
        final File testData = new File(fitFolder_, "data_test.dat");

        try (FileInputStream fIn = new FileInputStream(fitData))
        {
            _dataFit = RawFittingGrid.readFromStream(fIn, SimpleStatus.class, SimpleRegressor.class);
        }

        try (FileInputStream fIn = new FileInputStream(testData))
        {
            _dataTest = RawFittingGrid.readFromStream(fIn, SimpleStatus.class, SimpleRegressor.class);
        }
    }

    private ItemSettings generateSettings(final OptimizationTarget target_, final long seed_) throws IOException
    {
        final ItemClassifierSettings settings = ItemClassifierSettings.load(_settingsFile.getAbsolutePath());
        final ItemSettings itemSettings = settings.getSettings().toBuilder()
                .setTarget(target_).setRand(seed_).build();

        return itemSettings;
    }

    private PrintStream generatePrinter(final String filename_) throws IOException
    {
        final File outputFolder = new File(_fitFolder, "output");
        outputFolder.mkdirs();
        final File outputFile = new File(outputFolder, filename_);

        if (outputFile.exists())
        {
            // File exists, we will skip this stage.
            System.out.println("Skipping stage: " + outputFile.getAbsolutePath());
            return null;
        }

        final OutputStream oStream = new FileOutputStream(outputFile);
        final PrintStream printer = new PrintStream(oStream);
        return printer;
    }

    private File generateFitFile(final long seed, final OptimizationTarget target_)
    {
        final String suffix = "_" + target_.name();
        final String outputPath = "params_" + Long.toHexString(seed) + suffix + ".dat";
        final File modelFolder = new File(_fitFolder, "models");
        modelFolder.mkdirs();
        final File outputFile = new File(modelFolder, outputPath);
        return outputFile;
    }

    public void generateFit(final long seed, final OptimizationTarget target_, final int maxParams_) throws IOException,
            ConvergenceException
    {
        final File outputFile = generateFitFile(seed, target_);

        if (outputFile.exists())
        {
            System.out.println("Fit already exists: " + outputFile.getAbsolutePath());
            return;
        }

        final ItemClassifierSettings settings = ItemClassifierSettings.load(_settingsFile.getAbsolutePath());

        final ItemFitter<SimpleStatus, SimpleRegressor, StandardCurveType> fitter = new ItemFitter<>(
                settings.getFactory(),
                settings.getIntercept(), _dataFit, generateSettings(target_, seed));

        final int maxParams = maxParams_;

        final int usedParams = fitter.getBestParameters().getEffectiveParamCount();
        final int remainingParams = maxParams - usedParams;

        fitter.fitModel(settings.getNonCurveRegressors(), settings.getCurveRegressors(), remainingParams, false);
        final FitResult<SimpleStatus, SimpleRegressor, StandardCurveType> fitResult =
                fitter.getChain().getLatestResults();


        try (final FileOutputStream fout = new FileOutputStream(outputFile))
        {
            fitResult.writeToStream(fout);
        }
    }

    public FitResult<SimpleStatus, SimpleRegressor, StandardCurveType> loadFit(final long seed,
                                                                               final OptimizationTarget target_)
            throws IOException
    {
        final File outputFile = generateFitFile(seed, target_);

        if (!outputFile.exists())
        {
            return null;
        }

        try (final FileInputStream fin = new FileInputStream(outputFile))
        {
            return FitResult.readFromStream(fin, SimpleStatus.class, SimpleRegressor.class, StandardCurveType.class);
        }
    }

    public void generateFits(final int numFits_, final int maxParams_) throws Exception
    {
        final List<OptimizationTarget> targets = Arrays.asList(new OptimizationTarget[]{OptimizationTarget.ENTROPY,
                OptimizationTarget.ICE2, OptimizationTarget.ICE_STABLE_B, OptimizationTarget.ICE,
                OptimizationTarget.ICE_B,
                OptimizationTarget.ICE_RAW});

        for (int i = 0; i < numFits_; i++)
        {
            final long thisSeed = _startingSeed + i;

            for (final OptimizationTarget target : targets)
            {
                generateFit(thisSeed, target, maxParams_);
            }
        }

    }

    private List<FitResult<SimpleStatus, SimpleRegressor, StandardCurveType>> getAllParams(final int numFits_)
            throws IOException
    {
        List<FitResult<SimpleStatus, SimpleRegressor, StandardCurveType>> output = new ArrayList<>();

        final List<OptimizationTarget> targets = Arrays.asList(new OptimizationTarget[]{OptimizationTarget.ENTROPY,
                OptimizationTarget.ICE2, OptimizationTarget.ICE_STABLE_B, OptimizationTarget.ICE,
                OptimizationTarget.ICE_RAW});

        for (int i = 0; i < numFits_; i++)
        {
            final long thisSeed = _startingSeed + i;

            for (final OptimizationTarget target : targets)
            {
                final FitResult<SimpleStatus, SimpleRegressor, StandardCurveType> result = loadFit(thisSeed, target);

                if (null == result)
                {
                    continue;
                }

                output.add(result);

                FitResult<SimpleStatus, SimpleRegressor, StandardCurveType> prev = result.getPrev();

                while (prev != null)
                {
                    output.add(prev);
                    prev = prev.getPrev();
                }
            }
        }

        return Collections.unmodifiableList(output);
    }

    public void examineGradients(final List<FitResult<SimpleStatus, SimpleRegressor, StandardCurveType>> paramList,
                                 final PrintStream printer_)
            throws IOException
    {
        final EntropyCalculator<SimpleStatus, SimpleRegressor, StandardCurveType> entropyCalc = new EntropyCalculator<>(
                _dataFit, generateSettings(OptimizationTarget.ENTROPY, _startingSeed));
        final EntropyCalculator<SimpleStatus, SimpleRegressor, StandardCurveType> ice3Calc = new EntropyCalculator<>(
                _dataFit, generateSettings(OptimizationTarget.ICE_STABLE_B, _startingSeed));
        final EntropyCalculator<SimpleStatus, SimpleRegressor, StandardCurveType> ice4Calc = new EntropyCalculator<>(
                _dataFit, generateSettings(OptimizationTarget.ICE, _startingSeed));
        final EntropyCalculator<SimpleStatus, SimpleRegressor, StandardCurveType> ice5Calc = new EntropyCalculator<>(
                _dataFit, generateSettings(OptimizationTarget.ICE_B, _startingSeed));

        printer_.println(
                "params, evfd_mag, evfd_cos, ice3fd_mag, ice3fd_cos, ice4fd_mag, ice4fd_cos, ice5fd_mag, " +
                        "ice5fd_cos, ice3adj_mag, ice3adj_cos, ice4adj_mag, ice4adj_cos, ice5adj_mag, ice5adj_cos");

        for (final FitResult<SimpleStatus, SimpleRegressor, StandardCurveType> next : paramList)
        {
            GradientResult entropyGrad = entropyCalc.computeGradients(next.getParams());
            GradientResult ice3Grad = ice3Calc.computeGradients(next.getParams());
            GradientResult ice4Grad = ice4Calc.computeGradients(next.getParams());
            GradientResult ice5Grad = ice5Calc.computeGradients(next.getParams());

            final double[] eg = entropyGrad.getGradient();
            final double[] fdeg = entropyGrad.getFdGradient();

            final double ice3Mag = MathTools.magnitude(ice3Grad.getGradientAdjustment()) / MathTools.magnitude(eg);
            final double ice4Mag = MathTools.magnitude(ice4Grad.getGradientAdjustment()) / MathTools.magnitude(eg);
            final double ice5Mag = MathTools.magnitude(ice5Grad.getGradientAdjustment()) / MathTools.magnitude(eg);

            final double[] ice3FdAdj = MathTools.subtract(ice3Grad.getFdGradient(), fdeg);
            final double[] ice3Adj = ice3Grad.getGradientAdjustment();
            final double ice3MagRel = MathTools.magnitude(ice3Adj) / MathTools.magnitude(ice3FdAdj);

            final double[] ice4FdAdj = MathTools.subtract(ice4Grad.getFdGradient(), fdeg);
            final double[] ice4Adj = ice4Grad.getGradientAdjustment();
            final double ice4MagRel = MathTools.magnitude(ice4Adj) / MathTools.magnitude(ice4FdAdj);

            final double[] ice5FdAdj = MathTools.subtract(ice5Grad.getFdGradient(), fdeg);
            final double[] ice5Adj = ice5Grad.getGradientAdjustment();
            final double ice5MagRel = MathTools.magnitude(ice5Adj) / MathTools.magnitude(ice5FdAdj);

            printer_.print(next.getParams().getEffectiveParamCount());
            printer_.print(", " + MathTools.magnitude(eg) / MathTools.magnitude(fdeg));
            printer_.print(", " + MathTools.cos(eg, fdeg));
            printer_.print(", " + ice3Mag);
            printer_.print(", " + MathTools.cos(ice3Grad.getGradient(), ice3Grad.getFdGradient()));
            printer_.print(", " + ice4Mag);
            printer_.print(", " + MathTools.cos(ice4Grad.getGradient(), ice4Grad.getFdGradient()));
            printer_.print(", " + ice5Mag);
            printer_.print(", " + MathTools.cos(ice5Grad.getGradient(), ice5Grad.getFdGradient()));

            printer_.print(", " + ice3MagRel);
            printer_.print(", " + MathTools.cos(ice3Adj, ice3FdAdj));
            printer_.print(", " + ice4MagRel);
            printer_.print(", " + MathTools.cos(ice4Adj, ice4FdAdj));
            printer_.print(", " + ice5MagRel);
            printer_.println(", " + MathTools.cos(ice5Adj, ice5FdAdj));
        }


    }

    public void examineCorrections(final List<FitResult<SimpleStatus, SimpleRegressor, StandardCurveType>> paramList,
                                   final PrintStream printer_)
            throws IOException
    {
        final ItemSettings entropySettings =
                generateSettings(OptimizationTarget.ENTROPY, _startingSeed).toBuilder()
                        .setComplexFitResults(true)
                        .build();
        final EntropyCalculator<SimpleStatus, SimpleRegressor, StandardCurveType> fitCalc = new EntropyCalculator<>(
                _dataFit, entropySettings);

        printer_.println("Total param count: " + paramList.size());

        printer_.println("paramCount, entropy, invCondNumberJ, invCondNumberI, ticSum, iceSum, " +
                "iceSum2, iceSum3");

        for (final FitResult<SimpleStatus, SimpleRegressor, StandardCurveType> next : paramList)
        {
            final FitResult<SimpleStatus, SimpleRegressor, StandardCurveType> nextFit = fitCalc
                    .computeFitResult(next.getParams(), null);

            printer_.print(nextFit.getParams().getEffectiveParamCount());
            printer_.print(", " + nextFit.getEntropy());
            printer_.print(", " + nextFit.getInvConditionNumberJ());
            printer_.print(", " + nextFit.getInvConditionNumberI());
            printer_.print(", " + nextFit.getTicSum());
            printer_.print(", " + nextFit.getIceSum());
            printer_.print(", " + nextFit.getIce2Sum());
            printer_.println(", " + nextFit.getIce3Sum());
        }


    }

    public void examineCost(final List<FitResult<SimpleStatus, SimpleRegressor, StandardCurveType>> paramList,
                            final PrintStream printer_)
            throws IOException
    {
        final ItemSettings entropySettings =
                generateSettings(OptimizationTarget.ENTROPY, _startingSeed).toBuilder().setzScoreCutoff(0.0)
                        .setAicCutoff(0.0)
                        .setL2Lambda(0.00001)
                        .build();
        final EntropyCalculator<SimpleStatus, SimpleRegressor, StandardCurveType> testCalc = new EntropyCalculator<>(
                _dataTest, entropySettings);
        final EntropyCalculator<SimpleStatus, SimpleRegressor, StandardCurveType> fitCalc = new EntropyCalculator<>(
                _dataFit, entropySettings);

        final List<OptimizationTarget> targets = Arrays.asList(
                OptimizationTarget.ICE_RAW,
                OptimizationTarget.L2,
                OptimizationTarget.ICE_B,
                OptimizationTarget.ICE);

        printer_.println("Total param count: " + paramList.size());


        printer_.println("target, paramCount, valueTime, gradientTime");

        double accumulator = 0.0;
        final int loopCount = 1;

        for (final FitResult<SimpleStatus, SimpleRegressor, StandardCurveType> next : paramList)
        {
            final long eStart = System.currentTimeMillis();

            final FitPointAnalyzer analyzer = new FitPointAnalyzer(entropySettings.getBlockSize(),
                    OptimizationTarget.ENTROPY, entropySettings);

            final EntropyCalculator<SimpleStatus, SimpleRegressor, StandardCurveType> mleCalc =
                    new EntropyCalculator<>(_dataTest, entropySettings);

            for (int i = 0; i < loopCount; i++)
            {
                final FitPoint point = mleCalc.generatePoint(next.getParams());
                accumulator += analyzer.computeObjective(point, point.getBlockCount());
            }
            final long end = System.currentTimeMillis();
            final long eElapsed = end - eStart;

            for (int i = 0; i < loopCount; i++)
            {
                final FitPoint point = mleCalc.generatePoint(next.getParams());
                accumulator += MathTools.magnitude(analyzer.getDerivative(point));
            }

            final long elapsed2 = System.currentTimeMillis() - end;

            printer_.print(OptimizationTarget.ENTROPY);
            printer_.print(", " + next.getParams().getEffectiveParamCount());
            printer_.print(", " + eElapsed);
            printer_.println(", " + elapsed2);

            for (final OptimizationTarget target : targets)
            {
                final long nStart = System.currentTimeMillis();

                final FitPointAnalyzer analyzer2 = new FitPointAnalyzer(entropySettings.getBlockSize(),
                        target,
                        entropySettings.toBuilder().setTarget(target).build());

                final EntropyCalculator<SimpleStatus, SimpleRegressor, StandardCurveType> reCalc =
                        new EntropyCalculator<>(_dataTest, entropySettings.toBuilder().setTarget(target).build());

                for (int i = 0; i < loopCount; i++)
                {
                    final FitPoint point = reCalc.generatePoint(next.getParams());
                    accumulator += analyzer2.computeObjective(point, point.getBlockCount());
                }

                final long nEnd = System.currentTimeMillis();
                final long nElapsed = nEnd - nStart;

                for (int i = 0; i < loopCount; i++)
                {
                    final FitPoint point = reCalc.generatePoint(next.getParams());
                    accumulator += MathTools.magnitude(analyzer2.getDerivative(point));
                }

                final long nElapsed2 = System.currentTimeMillis() - nEnd;

                printer_.print(target);
                printer_.print(", " + next.getParams().getEffectiveParamCount());
                printer_.print(", " + nElapsed);
                printer_.println(", " + nElapsed2);
            }
        }
    }

    public ItemParameters<SimpleStatus, SimpleRegressor, StandardCurveType> getBestFit(final int fitCount_)
            throws Exception
    {
        final List<FitResult<SimpleStatus, SimpleRegressor, StandardCurveType>> paramList = getAllParams(fitCount_);

        ItemParameters<SimpleStatus, SimpleRegressor, StandardCurveType> bestParams = paramList.get(0).getParams();
        double bestAic = Double.POSITIVE_INFINITY;

        final ItemSettings entropySettings =
                generateSettings(OptimizationTarget.ENTROPY, _startingSeed).toBuilder().setzScoreCutoff(0.0)
                        .setAicCutoff(0.0)
                        .setL2Lambda(0.00001)
                        .build();

        final EntropyCalculator<SimpleStatus, SimpleRegressor, StandardCurveType> fitCalc = new EntropyCalculator<>(
                _dataFit, entropySettings);

        for (final FitResult<SimpleStatus, SimpleRegressor, StandardCurveType> next : paramList)
        {
            final double nextAic = (_dataFit.size() * fitCalc.computeEntropy(next.getParams()).getEntropyMean()) + next
                    .getParams().getEffectiveParamCount();

            if (nextAic < bestAic)
            {
                bestParams = next.getParams();
                bestAic = nextAic;
                System.out.println("Improved model[" + bestAic + "]: " + bestParams.getEffectiveParamCount());
            }
        }


        System.out.println(
                "Identified best model[" + bestAic + "][" + bestParams.getEffectiveParamCount() + "]: " + bestParams);

        return bestParams;
    }


    public List<FitResult<SimpleStatus, SimpleRegressor, StandardCurveType>> generateOverfit(final int fitCount_)
            throws Exception
    {
        final ItemParameters<SimpleStatus, SimpleRegressor, StandardCurveType> bestParams = getBestFit(fitCount_);
        final ItemClassifierSettings settings = ItemClassifierSettings.load(_settingsFile.getAbsolutePath());
        final ItemSettings entropySettings =
                generateSettings(OptimizationTarget.ENTROPY, _startingSeed).toBuilder().setzScoreCutoff(0.0)
                        .setAicCutoff(0.0)
                        .setL2Lambda(0.00001)
                        .build();

        final EntropyCalculator<SimpleStatus, SimpleRegressor, StandardCurveType> fitCalc = new EntropyCalculator<>(
                _dataFit, entropySettings);


        Map<SimpleRegressor, QuantileBreakdown> quantiles = new TreeMap<>();

        for (SimpleRegressor curveReg : settings.getCurveRegressors())
        {
            final ItemRegressorReader reader = _dataFit.getRegressorReader(curveReg);
            QuantileBreakdown approx = QuantileBreakdown.buildApproximation(reader);
            quantiles.put(curveReg, approx);
        }


        final int maxQuantileSteps = 10;
        final int minQuantileSteps = 3;
        final double smallChange = 1.0;


        final StandardCurveFactory factory = StandardCurveType.LOGISTIC.getFactory();
        final List<FitResult<SimpleStatus, SimpleRegressor, StandardCurveType>> expandedList = new ArrayList<>();
        expandedList.add(fitCalc.computeFitResult(bestParams, null));

        System.out.println("Curve regressor count: " + settings.getCurveRegressors().size());

        for (int i = minQuantileSteps; i <= maxQuantileSteps; i++)
        {
            ItemParameters<SimpleStatus, SimpleRegressor, StandardCurveType> baseline = bestParams;

            final ItemFitter<SimpleStatus, SimpleRegressor, StandardCurveType> fitter =
                    new ItemFitter<>(baseline,
                            _dataFit,
                            ItemSettings.newBuilder().setTarget(OptimizationTarget.ENTROPY).setRand(_startingSeed)
                                    .setzScoreCutoff(0.0).setAicCutoff(0.0)
                                    .build());


            final double interceptChange = 0.0; //-smallChange / i;

            for (SimpleRegressor curveReg : settings.getCurveRegressors())
            {
                QuantileBreakdown rawQuantiles = quantiles.get(curveReg);

                if (rawQuantiles.getSize() < i)
                {
                    System.out.println("Not enough quantiles[" + i + "]: " + curveReg);
                    continue;
                }

                QuantileBreakdown reduced = rawQuantiles.rebucket(i);

                for (int j = 1; j < i - 1; j++)
                {
                    final double mean = reduced.getBucketMean(j);
                    final double spread = 0.5 * (reduced.getBucketMean(j + 1) - reduced.getBucketMean(j - 1));

                    if (Double.isNaN(spread) || spread == 0.0)
                    {
                        continue;
                    }

                    ItemCurve<StandardCurveType> logisticCurve = factory.generateCurve(StandardCurveType.LOGISTIC, 0,
                            new double[]{mean, 1.0 / spread});

                    ItemCurve<StandardCurveType> gaussianCurve = factory.generateCurve(StandardCurveType.GAUSSIAN, 0,
                            new double[]{mean, spread});

                    ItemCurveParams<SimpleRegressor, StandardCurveType> logisticParams = new ItemCurveParams<>(
                            -interceptChange,
                            smallChange,
                            curveReg,
                            logisticCurve);

                    ItemCurveParams<SimpleRegressor, StandardCurveType> gaussianParams =
                            new ItemCurveParams<>(-interceptChange,
                                    smallChange,
                                    curveReg,
                                    gaussianCurve);

                    for (SimpleStatus toStatus : bestParams.getStatus().getFamily().getMembers())
                    {
                        if (toStatus.equals(bestParams.getStatus()))
                        {
                            continue;
                        }

                        baseline = baseline.addBeta(logisticParams, toStatus);
                        baseline = baseline.addBeta(gaussianParams, toStatus);
                    }
                }

                fitter.pushParameters("Expanded[" + i + "][" + curveReg + "]", baseline);

                // Fit coefficients first, because they may be very far from reasonable. Then fit everything.
                fitter.fitCoefficients();
                FitResult<SimpleStatus, SimpleRegressor, StandardCurveType> result = fitter.fitAllParameters();
                baseline = result.getParams();
                expandedList.add(result);
            }
        }

        return Collections.unmodifiableList(expandedList);
    }


    public void examineResiduals2(
            final List<FitResult<SimpleStatus, SimpleRegressor, StandardCurveType>> paramList,
            final PrintStream printer_) throws Exception
    {
        final ItemSettings entropySettings =
                generateSettings(OptimizationTarget.ENTROPY, _startingSeed).toBuilder().setzScoreCutoff(0.0)
                        .setAicCutoff(0.0)
                        .setL2Lambda(0.00001)
                        .build();
        final EntropyCalculator<SimpleStatus, SimpleRegressor, StandardCurveType> testCalc = new EntropyCalculator<>(
                _dataTest, entropySettings);
        final EntropyCalculator<SimpleStatus, SimpleRegressor, StandardCurveType> fitCalc = new EntropyCalculator<>(
                _dataFit, entropySettings);

        final List<OptimizationTarget> targets = Arrays.asList(
                OptimizationTarget.ENTROPY,
                OptimizationTarget.ICE_B,
                OptimizationTarget.ICE);

        printer_.println("Total param count: " + paramList.size());


        printer_.println("target, paramCount, fitTime, entropy, invCondNumberJ, invCondNumberI, ticSum, iceSum, " +
                "iceSum2, iceSum3, testEntropy");

        List<FitResult<SimpleStatus, SimpleRegressor, StandardCurveType>> reverseList =
                new ArrayList<>(paramList);
        Collections.reverse(reverseList);


        for (final FitResult<SimpleStatus, SimpleRegressor, StandardCurveType> next : reverseList)
        {
            ItemFitter<SimpleStatus, SimpleRegressor, StandardCurveType> fitter =
                    new ItemFitter<>(next.getParams(), _dataFit, entropySettings);

            final FitResult<SimpleStatus, SimpleRegressor, StandardCurveType> mle = fitter.fitAllParameters();

            for (final OptimizationTarget target : targets)
            {
                final long nStart = System.currentTimeMillis();
                ItemFitter<SimpleStatus, SimpleRegressor, StandardCurveType> refitter =
                        new ItemFitter<>(mle.getParams(), _dataFit,
                                entropySettings.toBuilder().setTarget(target).build());

                refitter.fitCoefficients();
                final FitResult<SimpleStatus, SimpleRegressor, StandardCurveType> nextFit = refitter
                        .fitAllParameters();
                final long nElapsed = System.currentTimeMillis() - nStart;

                printer_.print(target);
                printer_.print(", " + nextFit.getParams().getEffectiveParamCount());
                printer_.print(", " + nElapsed);
                printer_.print(", " + fitCalc.computeEntropy(nextFit.getParams()).getEntropyMean());
                printer_.print(", " + nextFit.getInvConditionNumberJ());
                printer_.print(", " + nextFit.getInvConditionNumberI());
                printer_.print(", " + nextFit.getTicSum());
                printer_.print(", " + nextFit.getIceSum());
                printer_.print(", " + nextFit.getIce2Sum());
                printer_.print(", " + nextFit.getIce3Sum());
                printer_.println(", " + testCalc.computeEntropy(nextFit.getParams()).getEntropyMean());
            }
        }
    }


    public void printResults(final int numFits_, final PrintStream printer_) throws IOException
    {
        final ItemSettings entropySettings = generateSettings(OptimizationTarget.ENTROPY, _startingSeed);

        final List<OptimizationTarget> targets = Arrays.asList(new OptimizationTarget[]{OptimizationTarget.ENTROPY,
                OptimizationTarget.ICE2, OptimizationTarget.ICE_STABLE_B, OptimizationTarget.ICE,
                OptimizationTarget.ICE_B,
                OptimizationTarget.ICE_RAW});

        final EntropyCalculator<SimpleStatus, SimpleRegressor, StandardCurveType> fitCalc = new EntropyCalculator<>(
                _dataFit, entropySettings);
        final EntropyCalculator<SimpleStatus, SimpleRegressor, StandardCurveType> testCalc =
                new EntropyCalculator<>(_dataTest, entropySettings);

        printer_.println("seed, target, params, fitEntropy, recalcEntropy, testEntropy");

        for (int i = 0; i < numFits_; i++)
        {
            final long thisSeed = _startingSeed + i;

            for (final OptimizationTarget target : targets)
            {
                final FitResult<SimpleStatus, SimpleRegressor, StandardCurveType> result = loadFit(thisSeed, target);

                if (null == result)
                {
                    printer_.println(thisSeed + ", " + target + ", null, null, null, null");
                    continue;
                }

                printer_.print(thisSeed + ", " + target + ", ");
                printer_.print(result.getParams().getEffectiveParamCount() + ", " + result.getEntropy());

                final FitResult<SimpleStatus, SimpleRegressor, StandardCurveType> recalcFit =
                        fitCalc.computeFitResult(result.getParams(), result);
                final FitResult<SimpleStatus, SimpleRegressor, StandardCurveType> recalcTest =
                        testCalc.computeFitResult(result.getParams(), result);


                printer_.println(", " + recalcFit.getEntropy() + ", " + recalcTest.getEntropy());
            }
        }


    }


    public static void main(final String[] args) throws Exception
    {
        final int maxParams = 30;
        final int numFits = 20;

        ItemFitGenerator gen = new ItemFitGenerator(
                new File("/Users/tyler/sync-workspace/code/item_test"));
                //new File("./"));

        //gen.convertFits();
        gen.generateFits(numFits, maxParams);


        try (final PrintStream printer = gen.generatePrinter("results.csv"))
        {
            if (printer != null)
            {
                gen.printResults(numFits, printer);
            }
        }


        final List<FitResult<SimpleStatus, SimpleRegressor, StandardCurveType>> paramList = gen.getAllParams(numFits);

        try (final PrintStream printer = gen.generatePrinter("corrections.csv"))
        {
            if (printer != null)
            {
                gen.examineCorrections(paramList, printer);
            }
        }

        try (final PrintStream printer = gen.generatePrinter("residuals.csv"))
        {
            if (printer != null)
            {
                gen.examineResiduals2(paramList, printer);
            }
        }

        try (final PrintStream printer = gen.generatePrinter("cost.csv"))
        {
            if (printer != null)
            {
                gen.examineCost(paramList, printer);
            }
        }

        try (final PrintStream printer = gen.generatePrinter("gradients.csv"))
        {
            if (printer != null)
            {
                gen.examineGradients(paramList, printer);
            }
        }

        try (final PrintStream printer = gen.generatePrinter("overfit_residuals.csv"))
        {
            if (printer != null)
            {
                final List<FitResult<SimpleStatus, SimpleRegressor, StandardCurveType>> overfit = gen
                        .generateOverfit(numFits);
                gen.examineResiduals2(overfit, printer);
            }
        }

        System.out.println("Done.");
    }
}
