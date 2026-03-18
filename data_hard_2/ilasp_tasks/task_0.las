#constant(obj, o1).
#constant(obj, o11).
#constant(obj, o5).
#constant(obj, o6).
#constant(obj, o7).
#constant(obj, o8).

#modeh(d0(var(obj))).
#modeh(d1(var(obj))).
#modeh(d3(var(obj))).
#modeh(d4(var(obj))).
#modeh(d6(var(obj))).
#modeh(d7(var(obj))).

#modeb(1, d0(var(obj))).
#modeb(1, d0(var(obj)), (negative)).
#modeb(1, d1(var(obj))).
#modeb(1, d1(var(obj)), (negative)).
#modeb(1, d10(var(obj))).
#modeb(1, d10(var(obj)), (negative)).
#modeb(1, d11(var(obj))).
#modeb(1, d11(var(obj)), (negative)).
#modeb(1, d13(var(obj))).
#modeb(1, d13(var(obj)), (negative)).
#modeb(1, d2(var(obj))).
#modeb(1, d2(var(obj)), (negative)).
#modeb(1, d3(var(obj))).
#modeb(1, d3(var(obj)), (negative)).
#modeb(1, d4(var(obj))).
#modeb(1, d4(var(obj)), (negative)).
#modeb(1, d6(var(obj))).
#modeb(1, d6(var(obj)), (negative)).
#modeb(1, d7(var(obj))).
#modeb(1, d7(var(obj)), (negative)).
#modeb(1, d8(var(obj))).
#modeb(1, d8(var(obj)), (negative)).
#modeb(1, d9(var(obj))).
#modeb(1, d9(var(obj)), (negative)).

#pos(eg1, {
  d0(o11), d1(o7), d3(o7), d4(o7), d6(o7), d7(o7)
}, {
  d0(o1), d0(o5), d0(o6), d0(o7), d0(o8), d1(o1), d1(o11), d1(o5), d1(o6), d1(o8), d3(o1), d3(o11), d3(o5), d3(o6), d3(o8), d4(o1), d4(o11), d4(o5), d4(o6), d4(o8), d6(o11), d6(o5), d6(o6), d6(o8), d7(o1), d7(o11), d7(o8)
}, {
  d10(o11).
  d10(o8).
  d11(o11).
  d13(o7).
  d2(o7).
  d6(o1).
  d7(o5).
  d7(o6).
  d8(o6).
  d9(o6).
}).