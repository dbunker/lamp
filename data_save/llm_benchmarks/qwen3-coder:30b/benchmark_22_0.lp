d4(o0).
d1(o1).
d4(o4).
d0(o0).

d3(X) :- not d2(X), d4(X).
d2(X) :- not d0(X), d4(X).