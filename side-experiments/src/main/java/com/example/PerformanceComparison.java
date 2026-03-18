package com.example;

/**
 * Class containing two methods that perform similar operations
 * but with different performance characteristics.
 */
public class PerformanceComparison {

    /**
     * Optimized method: Uses StringBuilder for efficient string concatenation.
     * This method is faster for multiple concatenations.
     */
    public String optimizedMethod(int iterations) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < iterations; i++) {
            sb.append("Item").append(i).append(" ");
        }
        return sb.toString();
    }

    /**
     * Regressed method: Uses String concatenation in a loop.
     * This method is slower because it creates new String objects in each iteration.
     */
    public String regressedMethod(int iterations) {
        String result = "";
        for (int i = 0; i < iterations; i++) {
            result = result + "Item" + i + " ";
        }
        return result;
    }

    /**
     * Method with short string literal.
     * This method uses a short string literal to test if literal size affects performance.
     */
    public String methodWithShortLiteral(int iterations) {
        return "Short";
    }

    /**
     * Method with long string literal.
     * This method uses a long string literal to test if literal size affects performance.
     * The only difference from methodWithShortLiteral is the length of the string literal.
     */
    public String methodWithLongLiteral(int iterations) {
        return "ThisIsAVeryLongStringLiteralThatIsMuchLongerThanTheShortOneToTestIfLiteralSizeAffectsPerformance";
    }

    /**
     * Method with extra long string literal.
     * This method uses an extra long string literal to test if literal size affects performance.
     * The only difference from methodWithLongLiteral is the length of the string literal.
     */
    public String methodWithExtraLongLiteral(int iterations) {
        return "ThisIsAVeryLongStringLiteralThatIsMuchLongerThanTheShortOneToTestIfLiteralSizeAffectsPerformanceIsItReallyLongEnoughToTestIfItAffectsPerformanceYoureNotDoneYetButItsStillNotLongEnoughAndWeNeedToMakeItLongerToSeeIfItActuallyAffectsPerformanceButItsStillNotLongEnoughToSeeIfItActuallyAffectsPerformance";
    }
}
