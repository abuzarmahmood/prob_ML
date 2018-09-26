function [error] = ClassificationError(yHat, yTruth)
    error = sum(yHat(:) ~= yTruth(:)) / length(yHat);
end