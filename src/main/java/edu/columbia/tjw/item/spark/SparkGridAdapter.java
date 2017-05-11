/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package edu.columbia.tjw.item.spark;

import edu.columbia.tjw.item.ItemRegressor;
import edu.columbia.tjw.item.ItemRegressorReader;
import edu.columbia.tjw.item.ItemStatus;
import edu.columbia.tjw.item.data.ItemStatusGrid;
import edu.columbia.tjw.item.util.EnumFamily;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import org.apache.spark.ml.linalg.Vector;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;

/**
 *
 * @author tyler
 * @param <S>
 * @param <R>
 */
public class SparkGridAdapter<S extends ItemStatus<S>, R extends ItemRegressor<R>> implements ItemStatusGrid<S, R> {

    private final S _fromStatus;
    private final EnumFamily<R> _regFamily;
    private final int[] _toLabels;
    private final ItemRegressorReader[] _readers;

    public SparkGridAdapter(final Dataset<?> data_, final String labelColumn_,
            final String featureColumn_, final List<R> regressors_, final S fromStatus_, final R intercept_) {
        _fromStatus = fromStatus_;
        final int rowCount = (int) data_.count();
        final int regCount = regressors_.size();

        _regFamily = intercept_.getFamily();
        _readers = new ItemRegressorReader[_regFamily.size()];

        final Iterator<Row> rowForm = (Iterator<Row>) data_.toLocalIterator();

        final double[][] transposed = new double[regCount][rowCount];
        _toLabels = new int[rowCount];
        int pointer = 0;

        while (rowForm.hasNext()) {
            final Row next = rowForm.next();
            final Vector vec = (Vector) next.get(next.fieldIndex(featureColumn_));
            final Number label = (Number) next.getAs(next.fieldIndex(labelColumn_));

            if (vec.size() != regCount) {
                throw new IllegalArgumentException("Size mismatch.");
            }

            for (int i = 0; i < regCount; i++) {
                transposed[i][pointer] = vec.apply(i);
            }

            _toLabels[pointer] = label.intValue();
            pointer++;
        }

        for (int i = 0; i < regCount; i++) {
            final R reg = regressors_.get(i);
            final RawReader wrapped = new RawReader(transposed[i]);
            _readers[reg.ordinal()] = wrapped;
        }

        _readers[intercept_.ordinal()] = new InterceptReader(rowCount);
    }

    @Override
    public EnumFamily<S> getStatusFamily() {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    public int getStatus(int index_) {
        return _fromStatus.ordinal();
    }

    @Override
    public int getNextStatus(int index_) {
        return _toLabels[index_];
    }

    @Override
    public boolean hasNextStatus(int index_) {
        return true;
    }

    @Override
    public boolean hasRegressorReader(R field_) {
        return (null != _readers[field_.ordinal()]);
    }

    @Override
    public ItemRegressorReader getRegressorReader(R field_) {
        return _readers[field_.ordinal()];
    }

    @Override
    public int size() {
        return _toLabels.length;
    }

    @Override
    public EnumFamily<R> getRegressorFamily() {
        return _regFamily;
    }

    private static final class InterceptReader implements ItemRegressorReader {

        private final int _size;

        public InterceptReader(final int size_) {
            _size = size_;
        }

        @Override
        public double asDouble(int index_) {
            if (index_ < 0 || index_ >= _size) {
                throw new ArrayIndexOutOfBoundsException("Index out of bounds[0, " + _size + "): " + index_);
            }

            return 1.0;
        }

        @Override
        public int size() {
            return _size;
        }

    }

    private static final class RawReader implements ItemRegressorReader {

        private final double[] _data;

        public RawReader(final double[] data_) {
            _data = data_;
        }

        @Override
        public double asDouble(int index_) {
            return _data[index_];
        }

        @Override
        public int size() {
            return _data.length;
        }

    }
}
