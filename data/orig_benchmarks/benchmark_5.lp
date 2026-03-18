d0(o7).
d1(o4).
d7(o0).
d5(o0).
d9(o8).
d1(o1).
d3(o4).
d3(o9).
d6(o4).

d6(X) :- d9(X), d0(X), d2(X).
d8(X) :- d3(X), d6(X), d1(X).
d0(X) :- d6(X), d5(X), d1(X).
d5(X) :- d9(X), d0(X), d6(X).
d7(X) :- d6(X), d5(X), d1(X).
d8(X) :- d0(X), d5(X), d7(X).
d2(X) :- d9(X), d5(X), d7(X).
d0(X) :- d3(X), d7(X), d8(X).
d4(X) :- d2(X), d5(X), d7(X).
d9(X) :- d2(X), d3(X), d8(X).
d2(X) :- d9(X), d6(X), d1(X).