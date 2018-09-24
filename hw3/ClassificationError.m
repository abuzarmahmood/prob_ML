function [error] = ClassificationError(yHat, yTruth)
    error = 1 - sum(yHat(:) == yTruth(:)) / length(yHat);
end