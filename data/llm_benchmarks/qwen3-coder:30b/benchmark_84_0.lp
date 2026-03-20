d4(o0).
d1(o0).
d3(o8).
d2(o1).
d3(o0).

d5(X) :- d1(X), d4(X).
d2(X) :- d3(X), d1(X).
d0(X) :- d3(X), d1(X), d4(X).