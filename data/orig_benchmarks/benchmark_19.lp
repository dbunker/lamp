d1(o4).
d1(o0).
d2(o7).
d5(o0).
d4(o8).
d8(o8).

d9(X) :- d0(X), d3(X).
d7(X) :- d9(X), d6(X).
d5(X) :- d7(X), d1(X).
d4(X) :- d9(X), d8(X).
d1(X) :- d7(X), d4(X).
d0(X) :- d2(X), d3(X).
d0(X) :- d6(X), d1(X).
d8(X) :- d0(X), d2(X).
d3(X) :- d2(X), d7(X).
d7(X) :- d5(X), d4(X).
d6(X) :- d8(X), d4(X).