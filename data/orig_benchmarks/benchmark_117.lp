d0(o1).
d4(o1).
d1(o1).
d0(o4).

d0(X) :- d1(X), d3(X), d4(X).
d3(X) :- d1(X), d0(X), d4(X).