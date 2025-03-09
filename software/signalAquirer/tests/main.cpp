#include <gtest/gtest.h>

// A function to be tested
int add(int a, int b)
{
    return a + b;
}

// Test case
TEST(AdditionTest, HandlesPositiveNumbers)
{
    EXPECT_EQ(add(2, 3), 5);
    EXPECT_EQ(add(10, 20), 30);
}

TEST(AdditionTest, HandlesNegativeNumbers)
{
    EXPECT_EQ(add(-1, -1), -2);
    EXPECT_EQ(add(-10, 5), -5);
}

// Main function
int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
