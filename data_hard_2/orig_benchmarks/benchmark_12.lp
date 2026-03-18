d6(o6).
d0(o2).
d0(o3).
d7(o7).
d3(o8).
d15(o7).
d4(o1).
d17(o12).
d2(o2).
d14(o4).

d2(X) :- d4(X), not d15(X).
d9(X) :- not d4(X), d10(X).
d16(X) :- d17(X), not d13(X).
d16(X) :- d7(X), not d10(X).
d16(X) :- d5(X), not d10(X).
d4(X) :- d17(X), not d14(X).
d6(X) :- d2(X), not d9(X).
d14(X) :- d5(X), not d4(X).
d2(X) :- d3(X), not d5(X).
d10(X) :- not d2(X), d3(X).
d13(X) :- d6(X), not d7(X).
d3(X) :- d2(X), not d7(X).
d13(X) :- not d2(X), d15(X).
d5(X) :- d14(X), not d10(X).
d15(X) :- not d2(X), d10(X).