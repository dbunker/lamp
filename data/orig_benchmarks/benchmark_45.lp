d1(o9).
d0(o4).
d4(o4).
d5(o8).
d4(o8).
d0(o8).
d0(o1).
d3(o9).
d2(o6).

d0(X) :- d2(X), d1(X), d4(X).
d1(X) :- d0(X), d2(X), d4(X).
d3(X) :- d2(X), d5(X), d1(X).
d1(X) :- d0(X), d5(X), d4(X).
d0(X) :- d2(X), d3(X), d1(X).
d2(X) :- d3(X), d1(X), d4(X).
d2(X) :- d0(X), d5(X), d4(X).
d3(X) :- d0(X), d2(X), d4(X).
d2(X) :- d5(X), d1(X), d4(X).
d2(X) :- d0(X), d1(X), d4(X).
d4(X) :- d3(X), d5(X), d1(X).