d2(o3).
d4(o7).
d5(o7).
d1(o9).
d2(o8).
d7(o7).

d8(X) :- d5(X), d7(X), d4(X).
d8(X) :- d2(X), d6(X), d4(X).
d3(X) :- d6(X), d5(X), d8(X).
d0(X) :- d5(X), d6(X), d7(X).
d9(X) :- d0(X), d2(X), d8(X).
d1(X) :- d2(X), d3(X), d8(X).
d4(X) :- d9(X), d3(X), d1(X).
d1(X) :- d9(X), d3(X), d4(X).
d8(X) :- d9(X), d2(X), d6(X).
d6(X) :- d3(X), d5(X), d4(X).
d6(X) :- d9(X), d0(X), d4(X).