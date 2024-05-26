# Guilherme Azambuja

"""
This project aims to estimate land prices using fuzzy logic. By providing information such as the area of the land,
distance from the avenue, and distance from the beach, the system can generate an estimated price range.

To use the system, simply run this script. You will be prompted to enter the area of the land in square meters,
the distance from the avenue in kilometers, and the distance from the beach in kilometers.
The system will then provide an estimated price range for the land based on fuzzy logic principles.

Additionally, if you have access to the real price of the land, you can input it to compare it with the estimated price.

"""

from land_pricing import LandPricing


def main() -> None:
    land_pricing = LandPricing()

    while True:
        area = float(input("\nEnter the area(m²): "))
        dist_ave = float(input("Enter the distance from the avenue(km): "))
        dist_bch = float(input("Enter the distance from the beach(km): "))
        print()
        price = land_pricing.run(area, dist_ave, dist_bch)

        real_price = float(input("\nWhat is the real price? (type '0' to skip)\n-> "))

        if real_price != 0:
            LandPricing.compare_to_real_price(price, real_price)


if __name__ == '__main__':
    main()
