d9(o1).
d1(o1).
d5(o1).
d6(o0).
d2(o1).
d0(o3).

d2(X) :- d3(X), d7(X), d8(X).
d4(X) :- d9(X), d0(X), d5(X).
d1(X) :- d2(X), d3(X), d6(X).
d8(X) :- d5(X), d3(X), d7(X).
d3(X) :- d9(X), d2(X), d5(X).
d4(X) :- d9(X), d2(X), d1(X).
d2(X) :- d6(X), d7(X), d1(X).
d3(X) :- d9(X), d0(X), d2(X).
d3(X) :- d1(X), d5(X), d8(X).
d7(X) :- d0(X), d2(X), d5(X).
d4(X) :- d2(X), d5(X), d1(X).