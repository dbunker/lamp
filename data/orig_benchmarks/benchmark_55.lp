d5(o7).
d4(o0).
d2(o0).
d0(o9).
d3(o0).
d0(o2).

d1(X) :- d2(X), d3(X), d5(X).
d3(X) :- d0(X), d2(X), d1(X).
d0(X) :- d3(X), d5(X), d1(X).
d0(X) :- d2(X), d3(X), d5(X).
d0(X) :- d3(X), d1(X), d4(X).
d4(X) :- d3(X), d5(X), d1(X).
d4(X) :- d0(X), d3(X), d1(X).
d4(X) :- d2(X), d3(X), d5(X).
d1(X) :- d2(X), d3(X), d4(X).
d3(X) :- d2(X), d1(X), d4(X).
d3(X) :- d0(X), d5(X), d1(X).