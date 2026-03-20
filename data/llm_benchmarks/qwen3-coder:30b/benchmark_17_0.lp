d5(o7).
d6(o7).
d2(o9).
d9(o2).
d7(o7).
d1(o7).

d4(o9) :- not d6(o9), not d5(o9), not d7(o9), not d1(o9), not d2(o9).
d7(o9) :- not d6(o9), not d9(o9), not d5(o9), not d1(o9), not d2(o9).
d1(o9) :- not d6(o9), not d9(o9), not d5(o9), not d7(o9), not d2(o9).
d8(o7) :- not d6(o7), not d9(o7), not d7(o7), not d1(o7), not d5(o7), not d2(o7).
d3(o9) :- not d6(o9), not d5(o9), not d7(o9), not d1(o9), not d2(o9).
d0(o7) :- not d6(o7), not d9(o7), not d7(o7), not d1(o7), not d5(o7), not d2(o7).