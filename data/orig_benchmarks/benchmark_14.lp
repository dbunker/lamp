d5(o7).
d7(o4).
d4(o4).
d0(o8).
d8(o4).
d8(o9).

d4(X) :- d0(X), d5(X), d8(X).
d2(X) :- d7(X), d8(X), d4(X).
d1(X) :- d0(X), d6(X), d5(X).
d8(X) :- d2(X), d6(X), d5(X).
d7(X) :- d2(X), d5(X), d1(X).
d3(X) :- d7(X), d5(X), d4(X).
d6(X) :- d0(X), d5(X), d8(X).
d8(X) :- d0(X), d6(X), d7(X).
d6(X) :- d0(X), d3(X), d4(X).
d3(X) :- d7(X), d8(X), d4(X).
d5(X) :- d2(X), d3(X), d8(X).