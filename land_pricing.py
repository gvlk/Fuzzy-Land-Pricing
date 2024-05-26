# Guilherme Azambuja

import skfuzzy as fuzz
from skfuzzy import control as ctrl
from skfuzzy.control import Antecedent, Consequent, Rule
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple
from colorama import Fore


class AreaConstants:
    P00 = 180
    P20 = 250
    P40 = 300
    P60 = 320
    P80 = 360
    P100 = 455
    MIN = 100
    MAX = 500
    STEP = 1
    BELL_WIDTHS = (
        (P20 - P00) / 2,
        (P40 - P20) / 2,
        (P60 - P40) / 2,
        (P80 - P60) / 2,
        (P100 - P80) / 2
    )
    BELL_CENTERS = (
        P00 + BELL_WIDTHS[0],
        P20 + BELL_WIDTHS[1],
        P40 + BELL_WIDTHS[2],
        P60 + BELL_WIDTHS[3],
        P80 + BELL_WIDTHS[4]
    )


class PriceConstants:
    P00 = 42_400
    P20 = 132_000
    P40 = 160_000
    P60 = 180_000
    P80 = 238_600
    P100 = 400_000
    MIN = 20_000
    MAX = 500_000
    STEP = 1_000
    BELL_WIDTHS = (
        (P20 - P00) / 2,
        (P40 - P20) / 2,
        (P60 - P40) / 2,
        (P80 - P60) / 2,
        (P100 - P80) / 2
    )
    BELL_CENTERS = (
        P00 + BELL_WIDTHS[0],
        P20 + BELL_WIDTHS[1],
        P40 + BELL_WIDTHS[2],
        P60 + BELL_WIDTHS[3],
        P80 + BELL_WIDTHS[4]
    )


class DistAveConstants:
    P00 = 0.78
    P33 = 1.80
    P66 = 1.91
    P100 = 5.35
    AVERAGE = 2.30
    MIN = 0.60
    MAX = 5.50
    STEP = 0.05
    BELL_WIDTHS = (
        (P33 - P00) / 2,
        (P66 - P33) / 2,
        (P100 - P66) / 2
    )
    BELL_CENTERS = (
        P00 + BELL_WIDTHS[0],
        P33 + BELL_WIDTHS[1],
        P66 + BELL_WIDTHS[2]
    )


class DistBchConstants:
    P00 = 0.30
    P33 = 1.46
    P66 = 1.92
    P100 = 6.90
    AVERAGE = 1.98
    MIN = 0.20
    MAX = 7.00
    STEP = 0.05
    BELL_WIDTHS = (
        (P33 - P00) / 2,
        (P66 - P33) / 2,
        (P100 - P66) / 2
    )
    BELL_CENTERS = (
        P00 + BELL_WIDTHS[0],
        P33 + BELL_WIDTHS[1],
        P66 + BELL_WIDTHS[2]
    )


class LandPricing:
    BELL_SLOPE = 3

    def __init__(self) -> None:
        """
        Initialize the LandPricing class by creating antecedents, consequent, and fuzzy control rules.
        Also, initialize the control system and simulation, and visualize the membership functions.

        :return: None
        """

        self.area_ant, self.dist_ave_ant, self.dist_bch_ant = self.create_antecedents()
        self.price_con = self.create_consequent()

        self.rules = self.get_rules()
        self.land_pricing_ctrl = ctrl.ControlSystem(self.rules)
        self.land_pricing_sim = ctrl.ControlSystemSimulation(self.land_pricing_ctrl)

        self.area_ant.view()
        self.dist_ave_ant.view()
        self.dist_bch_ant.view()
        self.price_con.view()
        self.plot_area_vs_price()

    def create_antecedents(self) -> Tuple[Antecedent, Antecedent, Antecedent]:
        """
        Create and return the antecedent variables (inputs) for the fuzzy logic system.

        :return: Tuple[Antecedent, Antecedent, Antecedent]: The area, distance from the avenue, and distance from the beach antecedents.
        """

        area_ant = ctrl.Antecedent(
            np.arange(AreaConstants.MIN, AreaConstants.MAX, AreaConstants.STEP), "area"
        )
        dist_ave_ant = ctrl.Antecedent(
            np.arange(DistAveConstants.MIN, DistAveConstants.MAX, DistAveConstants.STEP), "dist_ave"
        )
        dist_bch_ant = ctrl.Antecedent(
            np.arange(DistBchConstants.MIN, DistBchConstants.MAX, DistBchConstants.STEP), "dist_bch"
        )

        area_ant["very_small"] = fuzz.gbellmf(
            area_ant.universe, AreaConstants.BELL_WIDTHS[0], self.BELL_SLOPE, AreaConstants.BELL_CENTERS[0]
        )
        area_ant["small"] = fuzz.gbellmf(
            area_ant.universe, AreaConstants.BELL_WIDTHS[1], self.BELL_SLOPE, AreaConstants.BELL_CENTERS[1]
        )
        area_ant["medium"] = fuzz.gbellmf(
            area_ant.universe, AreaConstants.BELL_WIDTHS[2], self.BELL_SLOPE, AreaConstants.BELL_CENTERS[2]
        )
        area_ant["large"] = fuzz.gbellmf(
            area_ant.universe, AreaConstants.BELL_WIDTHS[3], self.BELL_SLOPE, AreaConstants.BELL_CENTERS[3]
        )
        area_ant["very_large"] = fuzz.gbellmf(
            area_ant.universe, AreaConstants.BELL_WIDTHS[4], self.BELL_SLOPE, AreaConstants.BELL_CENTERS[4]
        )

        dist_ave_ant["close"] = fuzz.gbellmf(
            dist_ave_ant.universe, DistAveConstants.BELL_WIDTHS[0], self.BELL_SLOPE, DistAveConstants.BELL_CENTERS[0]
        )
        dist_ave_ant["moderate"] = fuzz.gbellmf(
            dist_ave_ant.universe, DistAveConstants.BELL_WIDTHS[1], self.BELL_SLOPE, DistAveConstants.BELL_CENTERS[1]
        )
        dist_ave_ant["far"] = fuzz.gbellmf(
            dist_ave_ant.universe, DistAveConstants.BELL_WIDTHS[2], self.BELL_SLOPE, DistAveConstants.BELL_CENTERS[2]
        )

        dist_bch_ant["close"] = fuzz.gbellmf(
            dist_bch_ant.universe, DistBchConstants.BELL_WIDTHS[0], self.BELL_SLOPE, DistBchConstants.BELL_CENTERS[0]
        )
        dist_bch_ant["moderate"] = fuzz.gbellmf(
            dist_bch_ant.universe, DistBchConstants.BELL_WIDTHS[1], self.BELL_SLOPE, DistBchConstants.BELL_CENTERS[1]
        )
        dist_bch_ant["far"] = fuzz.gbellmf(
            dist_bch_ant.universe, DistBchConstants.BELL_WIDTHS[2], self.BELL_SLOPE, DistBchConstants.BELL_CENTERS[2]
        )

        return area_ant, dist_ave_ant, dist_bch_ant

    def create_consequent(self) -> Consequent:
        """
        Create and return the consequent variable (output) for the fuzzy logic system.

        :return: Consequent: The price consequent.
        """

        price_con = ctrl.Consequent(np.arange(PriceConstants.MIN, PriceConstants.MAX, PriceConstants.STEP), "price")

        price_con["very_low"] = fuzz.gbellmf(
            price_con.universe, PriceConstants.BELL_WIDTHS[0], self.BELL_SLOPE, PriceConstants.BELL_CENTERS[0]
        )
        price_con["low"] = fuzz.gbellmf(
            price_con.universe, PriceConstants.BELL_WIDTHS[1], self.BELL_SLOPE, PriceConstants.BELL_CENTERS[1]
        )
        price_con["medium"] = fuzz.gbellmf(
            price_con.universe, PriceConstants.BELL_WIDTHS[2], self.BELL_SLOPE, PriceConstants.BELL_CENTERS[2]
        )
        price_con["high"] = fuzz.gbellmf(
            price_con.universe, PriceConstants.BELL_WIDTHS[3], self.BELL_SLOPE, PriceConstants.BELL_CENTERS[3]
        )
        price_con["very_high"] = fuzz.gbellmf(
            price_con.universe, PriceConstants.BELL_WIDTHS[4], self.BELL_SLOPE, PriceConstants.BELL_CENTERS[4]
        )

        return price_con

    def get_rules(self) -> Tuple[Rule, Rule, Rule, Rule, Rule]:
        """
        Defines the rules for the fuzzy system based on the antecedents and consequent.

        :return: Tuple[Rule, Rule, Rule, Rule, Rule]: Tuple containing the fuzzy rules.
        """

        return (
            ctrl.Rule(
                self.area_ant["very_small"] & (self.dist_ave_ant["far"] | self.dist_bch_ant["far"]),
                self.price_con["very_low"]
            ),
            ctrl.Rule(
                self.area_ant["small"] | self.dist_ave_ant["moderate"],
                self.price_con["low"]
            ),
            ctrl.Rule(
                self.area_ant["medium"] | self.dist_ave_ant["close"] | self.dist_bch_ant["close"],
                self.price_con["medium"]
            ),
            ctrl.Rule(
                self.area_ant["large"] | self.dist_ave_ant["close"] & self.dist_bch_ant["close"],
                self.price_con["high"]
            ),
            ctrl.Rule(
                self.area_ant["very_large"] & (self.dist_ave_ant["close"] & self.dist_bch_ant["close"]),
                self.price_con["very_high"]
            )
        )

    def plot_area_vs_price(self) -> None:
        """
        Plot the relationship between land area and estimated price.
        :return: None
        """

        areas = np.arange(AreaConstants.P00, AreaConstants.P100, 1)
        prices = []

        for area in areas:
            prices.append(self._run(area, DistAveConstants.AVERAGE, DistBchConstants.AVERAGE))

        fig, ax = plt.subplots()
        ax.plot(prices, areas, label="Fuzzy System Output")

        ax.xaxis.set_major_formatter('R${x:1.0f}')
        ax.set_xlabel("Price (R$)")
        ax.set_ylabel("Area (m²)")
        ax.set_title("Area vs Price with average distance from the avenue and the beach")
        ax.grid(True)
        ax.legend()

        plt.show()

    @staticmethod
    def compare_to_real_price(recommended: float, real: float) -> None:
        """
        Compare the recommended price with the real price and print the difference as a percentage.

        :param recommended: Recommended price.
        :param real: Real price.
        :return: None
        """

        difference = real - recommended
        percentage = (difference / real) * 100

        if difference > 0:
            message = "\nI got it {:.2f}% less than the real price.".format(percentage)
        else:
            message = "\nI got it {:.2f}% more than the real price.".format(abs(percentage))

        print(message)

    def _run(self, area: float, dist_ave: float, dist_bch: float) -> float:
        """
        Run the fuzzy control system simulation with the given input and return the estimated price.

        :param area: Area of the land.
        :param dist_ave: Distance from the avenue.
        :param dist_bch: Distance from the beach.
        :return: Estimated price of the land.
        """

        self.land_pricing_sim.input["area"] = area
        self.land_pricing_sim.input["dist_ave"] = dist_ave
        self.land_pricing_sim.input["dist_bch"] = dist_bch
        self.land_pricing_sim.compute()
        return self.land_pricing_sim.output["price"]

    def run(self, area: float, dist_ave: float, dist_bch: float) -> float:
        """
        Run the fuzzy control system simulation with the given input, print the estimated price, and return it.

        :param area: Area of the land.
        :param dist_ave: Distance from the avenue.
        :param dist_bch: Distance from the beach.
        :return: Estimated price of the land.
        """

        answer = self._run(area, dist_ave, dist_bch)
        formatted_answer = (
            "Recommended price:{} R$ {:,.2f}{}"
            .format(Fore.GREEN, answer, Fore.RESET)
            .replace(",", "v")
            .replace(".", ",")
            .replace("v", ".")
        )

        print("Area: {:.0f}m²".format(area))
        print("Distance from the avenue: {:.2f}km".format(dist_ave))
        print("Distance from the beach: {:.2f}km".format(dist_bch))
        print()
        print(formatted_answer)
        self.price_con.view(sim=self.land_pricing_sim)

        return answer
