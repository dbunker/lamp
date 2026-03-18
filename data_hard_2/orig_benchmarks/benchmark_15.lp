d13(o8).
d3(o7).
d5(o2).
d15(o8).
d6(o8).
d0(o3).
d7(o10).
d13(o6).
d1(o3).
d4(o4).
d6(o0).
d2(o1).
d11(o10).
d6(o9).
d3(o3).

d9(X) :- not d17(X), d4(X).
d1(X) :- not d13(X), d15(X).
d4(X) :- d7(X), not d10(X).
d4(X) :- d9(X), not d11(X).
d17(X) :- not d6(X), d13(X).
d1(X) :- not d0(X), d6(X).
d12(X) :- not d10(X), d4(X).
d10(X) :- not d1(X), d11(X).
d4(X) :- d7(X), not d11(X).
d9(X) :- not d6(X), d17(X).
d4(X) :- d9(X), not d7(X).
d5(X) :- d0(X), not d13(X).
d15(X) :- d7(X), not d17(X).
d2(X) :- not d1(X), d0(X).
d5(X) :- not d17(X), d3(X).