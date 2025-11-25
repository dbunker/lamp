d4(o4).
d3(o4).
d1(o4).
d2(o4).

d3(X) :- d1(X), d2(X), d0(X).
d3(X) :- d2(X), d0(X), d4(X).
d0(X) :- d1(X), d3(X), d4(X).